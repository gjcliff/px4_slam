import cv2
import numpy as np
import rclpy
import rerun as rr
import torch
from lightglue import SuperPoint
from px4_slam_interfaces.msg import MatchedPoints
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

torch.set_grad_enabled(False)
torch.set_float32_matmul_precision("high")


class MatchPoints(Node):
    latest_image_msg: Image | None
    latest_camera_info_msg: CameraInfo | None

    def __init__(self):
        super().__init__("match_points")
        rr.init("match_points")
        rr.spawn()
        rr.log("world", rr.ViewCoordinates.FRD, static=True)

        self._image_sub = self.create_subscription(
            Image, "camera/image_raw", self.image_callback, qos_profile_sensor_data
        )
        self._camera_info_sub = self.create_subscription(
            CameraInfo,
            "camera/camera_info",
            self.camera_info_callback,
            qos_profile_sensor_data,
        )
        self._matched_points_pub = self.create_publisher(
            MatchedPoints, "camera/matched_points", qos_profile_sensor_data
        )

        # superpoint for detection only, no matcher needed
        self.extractor = (
            SuperPoint(max_num_keypoints=128, detection_threshold=0.005, nms_radius=4)
            .eval()
            .cuda()
        )

        self.track_history: dict[
            int, list[tuple[int, int]]
        ] = {}  # track_id -> list of (x, y)
        self.max_history_len: int = 10

        # lk optical flow params
        self.lk_params = dict(
            winSize=(31, 31),
            maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        self.redetect_every: int = 20  # redetect with superpoint every N frames

        # track state
        self.prev_gray: np.ndarray | None = None
        self.prev_pts: np.ndarray | None = None  # (N, 1, 2) float32
        self.track_ids: list[int] = []
        self.track_lengths: dict[int, int] = {}
        self.next_track_id: int = 0
        self.min_track_length: int = 3

        self.latest_image_msg: Image | None = None
        self.frame_count: int = 0
        self.count: int = 0

    def ros_image_to_tensor(self, msg: Image) -> torch.Tensor:
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        if msg.encoding == "bgr8":
            tensor = tensor.flip(0)
        return tensor.cuda()

    def ros_image_to_gray(self, msg: Image) -> np.ndarray:
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        if msg.encoding == "bgr8":
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    def ros_image_to_numpy(self, msg: Image) -> np.ndarray:
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        if msg.encoding == "bgr8":
            img = img[..., ::-1]
        return img

    def detect_with_superpoint(self, msg: Image) -> np.ndarray:
        gpu_img = self.ros_image_to_tensor(msg)
        feats = self.extractor.extract(gpu_img)
        kps = feats["keypoints"][0].cpu().numpy()  # (N, 2)
        return kps.reshape(-1, 1, 2).astype(np.float32)

    def draw_track_ids(
        self, img: np.ndarray, pts: np.ndarray, track_ids: list[int]
    ) -> np.ndarray:
        img = img.copy()
        for pt, tid in zip(pts, track_ids):
            x, y = int(pt[0]), int(pt[1])
            cv2.putText(
                img, str(tid), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1
            )
        return img

    def redetect_and_merge(self, msg: Image) -> np.ndarray:
        new_pts = self.detect_with_superpoint(msg)  # (N, 1, 2)

        if self.prev_pts is None or len(self.prev_pts) == 0:
            # no existing tracks, just assign all new ids
            new_ids = list(range(self.next_track_id, self.next_track_id + len(new_pts)))
            self.next_track_id += len(new_pts)
            for tid in new_ids:
                self.track_lengths[tid] = 1
            self.track_ids = new_ids
            return new_pts

        existing = self.prev_pts.reshape(-1, 2)  # (M, 2)
        candidates = new_pts.reshape(-1, 2)  # (N, 2)

        merged_pts = list(existing)
        merged_ids = list(self.track_ids)

        for pt in candidates:
            dists = np.linalg.norm(existing - pt, axis=1)
            if dists.min() > 10:  # 10 pixel threshold — no nearby existing track
                tid = self.next_track_id
                self.next_track_id += 1
                self.track_lengths[tid] = 1
                merged_pts.append(pt)
                merged_ids.append(tid)

        self.track_ids = merged_ids
        return np.array(merged_pts).reshape(-1, 1, 2).astype(np.float32)

    def image_callback(self, msg: Image):
        gray = self.ros_image_to_gray(msg)

        # redetect with superpoint periodically or on first frame
        if self.prev_gray is None or self.frame_count % self.redetect_every == 0:
            self.prev_pts = self.redetect_and_merge(msg)
            if self.prev_pts is None or len(self.prev_pts) == 0:
                self.prev_gray = gray
                self.frame_count += 1
                return
            self.prev_gray = gray
            self.latest_image_msg = msg
            self.frame_count += 1
            return

        # track with lk optical flow
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_pts, None, **self.lk_params
        )

        good_mask = status.ravel() == 1
        pts0 = self.prev_pts[good_mask].reshape(-1, 2)
        pts1 = curr_pts[good_mask].reshape(-1, 2)
        track_ids = [tid for tid, ok in zip(self.track_ids, good_mask) if ok]

        # update track lengths
        for tid in track_ids:
            self.track_lengths[tid] = self.track_lengths.get(tid, 0) + 1

        # cull dead tracks
        self.track_lengths = {tid: self.track_lengths[tid] for tid in track_ids}

        # filter mature tracks for publishing
        mature_mask = np.array(
            [
                self.track_lengths.get(tid, 0) >= self.min_track_length
                for tid in track_ids
            ]
        )

        if len(track_ids) > 0 and any(mature_mask):
            mature_pts0 = pts0[mature_mask]
            mature_pts1 = pts1[mature_mask]
            mature_ids = [tid for tid, m in zip(track_ids, mature_mask) if m]

            msg_out = MatchedPoints()
            msg_out.header.stamp = self.get_clock().now().to_msg()
            msg_out.keyframe_id = self.count
            msg_out.points0_x = mature_pts0[:, 0].tolist()
            msg_out.points0_y = mature_pts0[:, 1].tolist()
            msg_out.points1_x = mature_pts1[:, 0].tolist()
            msg_out.points1_y = mature_pts1[:, 1].tolist()
            msg_out.track_ids = mature_ids
            self._matched_points_pub.publish(msg_out)

        for pt, tid in zip(pts1, track_ids):
            x, y = int(pt[0]), int(pt[1])
            if tid not in self.track_history:
                self.track_history[tid] = []
            self.track_history[tid].append((x, y))
            if len(self.track_history[tid]) > self.max_history_len:
                self.track_history[tid].pop(0)

        # cull dead tracks from history
        self.track_history = {
            tid: hist for tid, hist in self.track_history.items() if tid in track_ids
        }

        # draw tracks on current image
        img_curr = self.ros_image_to_numpy(msg).copy()
        for tid, hist in self.track_history.items():
            for i in range(1, len(hist)):
                cv2.line(img_curr, hist[i - 1], hist[i], (0, 255, 0), 1)
            cv2.circle(img_curr, hist[-1], 3, (0, 0, 255), -1)
            cv2.putText(img_curr, str(tid), hist[-1], cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)

        rr.set_time("frame", sequence=self.frame_count)
        rr.log("camera/tracks", rr.Image(img_curr), static=True)

        self.get_logger().info(
            f"tracked: {len(track_ids)}, mature: {int(mature_mask.sum()) if len(track_ids) > 0 else 0}, next_id: {self.next_track_id}"
        )

        # update state
        self.prev_gray = gray
        self.prev_pts = curr_pts[good_mask].reshape(-1, 1, 2)
        self.track_ids = track_ids
        self.latest_image_msg = msg
        self.frame_count += 1
        self.count += 1

    def camera_info_callback(self, msg: CameraInfo): ...


def main(args=None):
    rclpy.init(args=args)
    node = MatchPoints()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
