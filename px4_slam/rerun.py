import rclpy
from rclpy.node import Node

from std_msgs.msg import String

import rerun as rr

class RerunViewer(Node):

    def __init__(self):
        super().__init__('rerun_node')
        self.timer = self.create_timer(1.0, self.timer_callback)

        rr.init("rerun_gtsam")
        rr.spawn()

    def timer_callback(self):
        ...


def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = RerunViewer()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
