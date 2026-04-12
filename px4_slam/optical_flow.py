import cv2
import numpy as np
import rclpy
import rerun as rr
from px4_slam_interfaces.msg import MatchedPoints
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class OpticalFlowTracker(Node):
    def __init__(self):
        super().__init__("optical_flow_tracker")

        rr.init("optical_flow")
        rr.spawn()

        self._image_sub = self.create_subscription(
            Image, "camera/image_raw", self.image_callback, qos_profile_sensor_data
        )
        self._matched_points_pub = self.create_publisher(
            MatchedPoints, "camera/matched_points", qos_profile_sensor_data
        )

        # lucas-kanade optical flow params
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        # shi-tomasi corner detection params
        self.feature_params = dict(
            maxCorners=512,
            qualityLevel=0.01,
            minDistance=8,
            blockSize=7,
        )

        self.prev_gray: np.ndarray | None = None
        self.prev_pts: np.ndarray | None = None  # (N, 1, 2) float32
        self.track_ids: list[int] = []
        self.track_lengths: dict[int, int] = {}
        self.next_track_id = 0
        self.min_track_length = 3
        self.redetect_every = 10  # redetect features every N frames
        self.frame_count = 0
        self.count = 0

        self.latest_image_msg: Image | None = None

    def ros_image_to_gray(self, msg: Image) -> np.ndarray:
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        if msg.encoding == "bgr8":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return gray

    def detect_features(self, gray: np.ndarray) -> np.ndarray:
        pts = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
        return pts  # (N, 1, 2) float32

    def image_callback(self, msg: Image):
        self.get_logger().info("here")
        gray = self.ros_image_to_gray(msg)

        if (
            self.prev_gray is None
            or self.prev_pts is None
            or self.frame_count % self.redetect_every == 0
        ):
            # detect fresh features
            pts = self.detect_features(gray)
            if pts is None:
                self.prev_gray = gray
                return
            self.prev_pts = pts
            self.track_ids = [self.next_track_id + i for i in range(len(pts))]
            self.next_track_id += len(pts)
            for tid in self.track_ids:
                self.track_lengths[tid] = 1
            self.prev_gray = gray
            self.frame_count += 1
            self.latest_image_msg = msg
            return

        # track with lucas-kanade
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_pts, None, **self.lk_params
        )

        # filter good tracks
        good_mask = status.ravel() == 1
        pts0 = self.prev_pts[good_mask].reshape(-1, 2)
        pts1 = curr_pts[good_mask].reshape(-1, 2)
        track_ids = [tid for tid, ok in zip(self.track_ids, good_mask) if ok]

        # update track lengths
        for tid in track_ids:
            self.track_lengths[tid] = self.track_lengths.get(tid, 0) + 1

        # filter mature tracks
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

        # update state
        self.prev_gray = gray
        self.prev_pts = curr_pts[good_mask].reshape(-1, 1, 2)
        self.track_ids = track_ids
        self.track_lengths = {tid: self.track_lengths[tid] for tid in track_ids}
        self.frame_count += 1
        self.count += 1
        self.latest_image_msg = msg

        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        img_annotated = img.copy()
        for pt, tid in zip(pts1, track_ids):
            x, y = int(pt[0]), int(pt[1])
            cv2.putText(
                img_annotated,
                str(tid),
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (0, 255, 0),
                1,
            )

        rr.set_time("frame", sequence=self.frame_count)
        rr.log("camera/curr", rr.Image(img_annotated))
        rr.log("camera/curr/keypoints", rr.Points2D(pts1))
        if self.latest_image_msg is not None:
            img_prev = np.frombuffer(
                self.latest_image_msg.data, dtype=np.uint8
            ).reshape(self.latest_image_msg.height, self.latest_image_msg.width, -1)
            rr.log("camera/prev", rr.Image(img_prev))
            rr.log("camera/prev/keypoints", rr.Points2D(pts0))


def main(args=None):
    rclpy.init(args=args)
    node = OpticalFlowTracker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
