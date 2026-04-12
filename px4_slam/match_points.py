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
            SuperPoint(max_num_keypoints=512).eval().cuda()
        )  # load the extractor
        self.matcher = (
            (LightGlue(features="superpoint")).eval().cuda()
        )  # load the matcher
        self.get_logger().info(str(next(self.matcher.parameters()).device))
        self.get_logger().info(str(next(self.extractor.parameters()).device))
        # self.matcher.compile(mode="default")
        self._img_buffer_curr: torch.Tensor = torch.zeros(
            3, 480, 640, dtype=torch.float32
        ).pin_memory()
        self._img_buffer_prev: torch.Tensor = torch.zeros(
            3, 480, 640, dtype=torch.float32
        ).pin_memory()
        self.prev_feats = None
        self.latest_feats = None

        self.prev_image: Image | None = None

        self.latest_image_msg = None
        self.prev_timestamp = None
        self.count = 0

    def ros_image_to_tensor(self, msg: Image, buffer: torch.Tensor) -> torch.Tensor:
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        if msg.encoding == "bgr8":
            tensor = tensor.flip(0)  # flip channel dim instead of copying numpy array
        buffer.copy_(tensor)
        return buffer.cuda(non_blocking=True)

    def ros_image_to_numpy(self, msg: Image) -> np.ndarray:
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        if msg.encoding == "bgr8":
            img = img[..., ::-1]
        return img

    def add_vision_factor(
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

        msg = MatchedPoints()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.keyframe_id = self.count
        msg.points0_x = pts0[:, 0].tolist()
        msg.points0_y = pts0[:, 1].tolist()
        msg.points1_x = pts1[:, 0].tolist()
        msg.points1_y = pts1[:, 1].tolist()

        self._matched_points_pub.publish(msg)
        ender.record()
        torch.cuda.synchronize()
        self.get_logger().info(
            f"processing image took: {starter.elapsed_time(ender):.1f}ms"
        )

    def image_callback(self, msg: Image):

        gpu_img = self.ros_image_to_tensor(msg, self._img_buffer_curr)
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
            self.add_vision_factor()

            img_prev = self.ros_image_to_numpy(self.prev_image)
            img_curr = self.ros_image_to_numpy(msg)
            rr.log("camera/prev", rr.Image(img_prev), static=True)
            rr.log("camera/curr", rr.Image(img_curr), static=True)
            # log keypoints
            rr.log(
                "camera/prev/keypoints",
                rr.Points2D(self.prev_feats["keypoints"].cpu().numpy()),
                static=True,
            )
            rr.log(
                "camera/curr/keypoints",
                rr.Points2D(self.latest_feats["keypoints"].cpu().numpy()),
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
