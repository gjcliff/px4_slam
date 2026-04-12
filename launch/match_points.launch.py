from launch_ros.actions import Node

from launch import LaunchDescription


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="px4_slam",
                namespace="",
                executable="match_points",
                name="match_points",
            ),
            Node(
                package="ros_gz_bridge",
                executable="parameter_bridge",
                arguments=[
                    "/world/walls/model/x500_mono_cam_0/link/camera_link/sensor/imager/image@sensor_msgs/msg/Image@gz.msgs.Image",
                    "/world/walls/model/x500_mono_cam_0/link/camera_link/sensor/imager/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo",
                ],
                remappings=[
                    ("/world/walls/model/x500_mono_cam_0/link/camera_link/sensor/imager/image", "/camera/image_raw"),
                    ("/world/walls/model/x500_mono_cam_0/link/camera_link/sensor/imager/camera_info", "/camera/camera_info"),
                ],
                output="screen",
            ),
        ]
    )
