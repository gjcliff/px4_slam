import rclpy
from px4_msgs.msg import VehicleGlobalPosition, SensorImu
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data


class PX4Slam(Node):
    def __init__(self):
        super().__init__("px4_slam")

        self._imu_sub: rclpy.node.Subscription = self.create_subscription(
            SensorImu,
            "/px4_1/fmu/out/vehicle_imu",
            self.imu_callback,
            qos_profile=qos_profile_sensor_data,
        )
        self._global_position_sub: rclpy.node.Subscription = self.create_subscription(
            VehicleGlobalPosition,
            "/px4_1/fmu/out/vehicle_global_position",
            self.global_position_callback,
            qos_profile=qos_profile_sensor_data,
        )

    def imu_callback(self, msg: SensorImu):
        self.get_logger().info(f"got imu message: \n"
                               f"x: {msg.}")
    def global_position_callback(self, msg: VehicleGlobalPosition): ...


def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = PX4Slam()
    rclpy.spin(minimal_subscriber)

    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
