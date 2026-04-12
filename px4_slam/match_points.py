import cv2
import numpy as np
import rclpy
import rerun as rr
import torch
from lightglue import LightGlue, SuperPoint
from px4_slam_interfaces.msg import MatchedPoints
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

torch.set_grad_enabled(False)
torch.set_float32_matmul_precision("high")  # or 'medium' for more speed


class MatchPoints(Node):
    biasKey: int
    latest_image_msg: Image | None
    latest_camera_info_msg: CameraInfo | None

    def __init__(self):
        super().__init__("px4_slam")
        rr.init("match_points")
        rr.spawn()
        rr.log("world", rr.ViewCoordinates.FRD, static=True)
        self._image_sub: rclpy.node.Subscription = self.create_subscription(
            Image,
            "camera/image_raw",
            self.image_callback,
            qos_profile=qos_profile_sensor_data,
        )
        self._camera_info_sub: rclpy.node.Subscription = self.create_subscription(
            CameraInfo,
            "camera/camera_info",
            self.camera_info_callback,
            qos_profile=qos_profile_sensor_data,
        )
        self._matched_points_pub: rclpy.node.Publisher = self.create_publisher(
            MatchedPoints,
            "camera/matched_points",
            qos_profile=qos_profile_sensor_data,
        )

        # SuperPoint+LightGlue
        self.extractor = (
            SuperPoint(max_num_keypoints=512, detection_threshold=0.005, nms_radius=4)
            .eval()
            .cuda()
        )
        self.matcher = (
            LightGlue(features="superpoint", width_confidence=-1, depth_confidence=-1, filter_threshold=0.05)
            .eval()
            .cuda()
        )
        self.get_logger().info(str(next(self.matcher.parameters()).device))
        self.get_logger().info(str(next(self.extractor.parameters()).device))
        # self.matcher.compile(mode="default")
        self.prev_feats = None
        self.latest_feats = None

        self.track_length: dict[int, int] = {}  # track_id -> number of frames seen
        self.min_track_length: int = 20  # minimum frames before promoting to map point
        self.track_coasting: dict[int, int] = {}  # track_id -> frames since last seen
        self.max_coasting_frames: int = 10  # keep track alive for 3 missed frames
        self.active_tracks: dict[int, int] = {}
        self.next_track_id = 0

        self.prev_image: Image | None = None

        self.latest_image_msg = None
        self.prev_timestamp = None
        self.count = 0

    def ros_image_to_tensor(self, msg: Image) -> torch.Tensor:
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        if msg.encoding == "bgr8":
            tensor = tensor.flip(0)
        return tensor.cuda()

    def ros_image_to_numpy(self, msg: Image) -> np.ndarray:
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        if msg.encoding == "bgr8":
            img = img[..., ::-1]
        return img

    def assign_track_ids(self, match_indices: torch.Tensor) -> list[int]:
        prev_kp_to_track = {kp_idx: tid for tid, kp_idx in self.active_tracks.items()}

        new_active_tracks = {}
        track_ids = []
        matched_track_ids = set()

        for idx0, idx1 in match_indices.tolist():
            if idx0 in prev_kp_to_track:
                tid = prev_kp_to_track[idx0]
            else:
                tid = self.next_track_id
                self.next_track_id += 1
                self.track_length[tid] = 0

            new_active_tracks[tid] = idx1
            self.track_coasting[tid] = 0
            self.track_length[tid] = self.track_length.get(tid, 0) + 1
            matched_track_ids.add(tid)
            track_ids.append(tid)

        # keep unmatched tracks alive for a few frames
        for tid, kp_idx in self.active_tracks.items():
            if tid not in matched_track_ids:
                self.track_coasting[tid] = self.track_coasting.get(tid, 0) + 1
                if self.track_coasting[tid] <= self.max_coasting_frames:
                    new_active_tracks[tid] = kp_idx  # keep last known position

        # cull expired tracks
        self.track_coasting = {
            tid: c for tid, c in self.track_coasting.items() if tid in new_active_tracks
        }
        self.track_length = {
            tid: l for tid, l in self.track_length.items() if tid in new_active_tracks
        }

        self.active_tracks = new_active_tracks
        return track_ids

    def match_points_and_pub(
        self,
    ):
        assert self.prev_feats is not None
        assert self.latest_feats is not None
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()
        matches = self.matcher({"image0": self.prev_feats, "image1": self.latest_feats})
        match_indices = matches["matches"][0]  # (K, 2) tensor
        kp0 = self.prev_feats["keypoints"][0]  # (N, 2) tensor
        kp1 = self.latest_feats["keypoints"][0]  # (M, 2) tensor

        # index into keypoints using match indices
        pts0 = kp0[match_indices[:, 0]].cpu().numpy()  # (K, 2)
        pts1 = kp1[match_indices[:, 1]].cpu().numpy()  # (K, 2)

        track_ids = self.assign_track_ids(match_indices)

        self.get_logger().info(
            f"matches: {len(track_ids)}, new tracks: {sum(1 for tid in track_ids if self.track_length.get(tid, 0) == 1)}, active: {len(self.active_tracks)}"
        )
        self.get_logger().info(f"self.next_track_id: {self.next_track_id}")

        if len(track_ids) == 0:
            return pts0, pts1, track_ids

        mature_mask = np.array(
            [
                self.track_length.get(tid, 0) >= self.min_track_length
                for tid in track_ids
            ]
        )
        mature_pts0 = pts0[mature_mask]
        mature_pts1 = pts1[mature_mask]
        mature_track_ids = [tid for tid, m in zip(track_ids, mature_mask) if m]

        desc1 = self.latest_feats["descriptors"][0]  # (M, 256)
        matched_desc1 = desc1[match_indices[:, 1]]  # (K, 256)
        mature_desc1 = matched_desc1[mature_mask]  # filter to mature only

        msg = MatchedPoints()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.keyframe_id = self.count
        msg.points0_x = mature_pts0[:, 0].tolist()
        msg.points0_y = mature_pts0[:, 1].tolist()
        msg.points1_x = mature_pts1[:, 0].tolist()
        msg.points1_y = mature_pts1[:, 1].tolist()
        msg.track_ids = mature_track_ids
        msg.descriptors1 = mature_desc1.cpu().flatten().tolist()

        self._matched_points_pub.publish(msg)
        ender.record()
        torch.cuda.synchronize()
        # self.get_logger().info(
        #     f"processing image took: {starter.elapsed_time(ender):.1f}ms"
        # )
        self.count += 1

        return mature_pts0, mature_pts1, track_ids

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

    def image_callback(self, msg: Image):

        gpu_img = self.ros_image_to_tensor(msg)
        feats = self.extractor.extract(gpu_img)
        self.prev_feats = self.latest_feats
        self.latest_feats = feats
        self.prev_image = self.latest_image_msg
        self.latest_image_msg = msg

        if (
            self.prev_feats is not None
            and self.latest_feats is not None
            and self.prev_image is not None
        ):
            pts0, pts1, track_ids = self.match_points_and_pub()

            img_prev = self.ros_image_to_numpy(self.prev_image)
            img_curr = self.ros_image_to_numpy(msg)
            img_curr_annotated = self.draw_track_ids(img_curr, pts1, track_ids)
            img_prev_annotated = self.draw_track_ids(img_prev, pts1, track_ids)

            rr.log("camera/prev", rr.Image(img_prev_annotated), static=True)
            rr.log("camera/curr", rr.Image(img_curr_annotated), static=True)
            rr.log(
                "camera/prev/keypoints",
                rr.Points2D(pts0),
                static=True,
            )
            rr.log(
                "camera/curr/keypoints",
                rr.Points2D(pts1),
                static=True,
            )

    def camera_info_callback(self, msg: CameraInfo): ...


def main(args=None):
    rclpy.init(args=args)

    match_points = MatchPoints()
    rclpy.spin(match_points)

    match_points.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
