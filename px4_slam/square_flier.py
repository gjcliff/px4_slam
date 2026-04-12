import numpy as np
import rclpy
from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleLocalPosition,
)
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy


class SquareFlyer(Node):
    def __init__(self):
        super().__init__("square_flyer")

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._offboard_pub = self.create_publisher(
            OffboardControlMode, "fmu/in/offboard_control_mode", qos
        )
        self._setpoint_pub = self.create_publisher(
            TrajectorySetpoint, "fmu/in/trajectory_setpoint", qos
        )
        self._command_pub = self.create_publisher(
            VehicleCommand, "fmu/in/vehicle_command", qos
        )
        self._local_pos_sub = self.create_subscription(
            VehicleLocalPosition,
            "fmu/out/vehicle_local_position",
            self.local_pos_callback,
            qos,
        )

        self.altitude = -10.0
        self.side_length = 20.0
        self.cruise_speed = 3.0  # m/s
        self.yaw_rate = 0.5  # rad/s for turning at corners

        # waypoints: (x, y, z, yaw) — yaw in radians, NED frame
        # yaw: 0=north, pi/2=east, pi=south, -pi/2=west
        self.waypoints = [
            [self.side_length, 0.0,              self.altitude, 0.0],          # fly north
            [self.side_length, self.side_length, self.altitude, np.pi / 2],    # fly east
            [0.0,              self.side_length, self.altitude, np.pi],         # fly south
            [0.0,              0.0,              self.altitude, -np.pi / 2],    # fly west
        ]
        self.current_waypoint = 0
        self.waypoint_threshold = 1.5

        self.local_pos = None
        self.counter = 0

        self.timer = self.create_timer(0.1, self.timer_callback)

    def local_pos_callback(self, msg: VehicleLocalPosition):
        self.local_pos = msg

    def publish_offboard_control_mode(self):
        msg = OffboardControlMode()
        msg.position = False
        msg.velocity = True
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = self.get_clock().now().nanoseconds // 1000
        self._offboard_pub.publish(msg)

    def publish_trajectory_setpoint(self, vx, vy, vz, yaw=0.0, yawspeed=0.0):
        msg = TrajectorySetpoint()
        msg.position = [
            float("nan"),
            float("nan"),
            float("nan"),
        ]  # must be nan for velocity control
        msg.velocity = [float(vx), float(vy), float(vz)]
        msg.yaw = yaw
        msg.yawspeed = float(yawspeed)  # rad/s
        msg.timestamp = self.get_clock().now().nanoseconds // 1000
        self._setpoint_pub.publish(msg)

    def publish_vehicle_command(self, command, param1=0.0, param2=0.0):
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = float(param1)
        msg.param2 = float(param2)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = self.get_clock().now().nanoseconds // 1000
        self._command_pub.publish(msg)

    def arm(self):
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0
        )
        self.get_logger().info("arming")

    def set_offboard_mode(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
        self.get_logger().info("switching to offboard mode")

    def compute_velocity(self) -> tuple[float, float, float]:
        if self.local_pos is None:
            return 0.0, 0.0, 0.0
        wp = self.waypoints[self.current_waypoint]
        dx = wp[0] - self.local_pos.x
        dy = wp[1] - self.local_pos.y
        dz = wp[2] - self.local_pos.z
        dist = np.sqrt(dx**2 + dy**2 + dz**2)
        if dist < 0.1:
            return 0.0, 0.0, 0.0
        # scale velocity by distance, cap at cruise speed
        scale = min(self.cruise_speed / dist, self.cruise_speed)
        return dx * scale, dy * scale, dz * scale

    def reached_waypoint(self) -> bool:
        if self.local_pos is None:
            return False
        wp = self.waypoints[self.current_waypoint]
        dist = np.sqrt(
            (self.local_pos.x - wp[0]) ** 2
            + (self.local_pos.y - wp[1]) ** 2
            + (self.local_pos.z - wp[2]) ** 2
        )
        return dist < self.waypoint_threshold

    def timer_callback(self):
        wp = self.waypoints[self.current_waypoint]
        vx, vy, vz = self.compute_velocity()

        # compute yawspeed to turn toward waypoint heading
        if self.local_pos is not None:
            target_yaw = wp[3]
            current_yaw = self.local_pos.heading  # radians
            yaw_err = np.arctan2(
                np.sin(target_yaw - current_yaw), np.cos(target_yaw - current_yaw)
            )
            yawspeed = np.clip(yaw_err * 1.0, -self.yaw_rate, self.yaw_rate)
        else:
            yawspeed = 0.0

        self.publish_offboard_control_mode()
        self.publish_trajectory_setpoint(vx, vy, vz, target_yaw, yawspeed)

        if self.counter == 10:
            self.set_offboard_mode()
            self.arm()

        if self.counter > 10 and self.reached_waypoint():
            self.current_waypoint = (self.current_waypoint + 1) % len(self.waypoints)
            self.get_logger().info(f"waypoint reached, next: {self.current_waypoint}")

        self.counter += 1


def main(args=None):
    rclpy.init(args=args)
    node = SquareFlyer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
