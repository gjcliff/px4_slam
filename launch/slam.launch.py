from launch_ros.actions import Node

from launch import LaunchDescription


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="px4_slam",
                namespace="",
                executable="gtsam",
                name="state_estimation",
            ),
            Node(
                package="px4_slam",
                namespace="",
                executable="super_flow",
                name="super_flow",
            ),
            Node(
                package="ros_gz_bridge",
                executable="parameter_bridge",
                arguments=[
                    "/world/baylands/model/x500_mono_cam_0/link/camera_link/sensor/imager/image@sensor_msgs/msg/Image@gz.msgs.Image",
                    "/world/baylands/model/x500_mono_cam_0/link/camera_link/sensor/imager/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo",
                ],
                remappings=[
                    ("/world/baylands/model/x500_mono_cam_0/link/camera_link/sensor/imager/image", "/camera/image_raw"),
                    ("/world/baylands/model/x500_mono_cam_0/link/camera_link/sensor/imager/camera_info", "/camera/camera_info"),
                ],
                output="screen",
            ),
        ]
    )
