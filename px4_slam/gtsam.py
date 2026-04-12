import gtsam
import numpy as np
import rclpy
import rerun as rr
from gtsam.symbol_shorthand import B, V, X
from px4_msgs.msg import SensorCombined, SensorGps, VehicleMagnetometer
from px4_slam_interfaces.msg import MatchedPoints
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image


class PX4Slam(Node):
    biasKey: int
    biasNoise: gtsam.noiseModel.Isotropic
    pim: gtsam.PreintegratedImuMeasurements
    latest_gps_msg: SensorGps | None = None
    latest_mag_msg: VehicleMagnetometer | None = None
    latest_matched_points: MatchedPoints | None = None
    latest_camera_info_msg: CameraInfo | None = None
    ref_sin_lat: float | None = None
    ref_cos_lat: float | None = None
    ref_lat: float | None = None
    ref_lon: float | None = None
    ref_alt: float | None = None
    prev_timestamp: int | None = None
    count: int = 0

    K: gtsam.Cal3_S2 | None = None
    pixelNoise: gtsam.noiseModel.Isotropic | None = None

    def __init__(self):
        super().__init__("px4_slam")
        rr.init("state_estimation")
        rr.spawn()
        rr.log("world", rr.ViewCoordinates.FRD, static=True)
        rr.log("world/drone", rr.TransformAxes3D(axis_length=1.0), static=True)
        self.trajectory: list[list[float]] = []

        self._imu_sub: rclpy.node.Subscription = self.create_subscription(
            SensorCombined,
            "fmu/out/sensor_combined",
            self.imu_callback,
            qos_profile=qos_profile_sensor_data,
        )
        self._gps_sub: rclpy.node.Subscription = self.create_subscription(
            SensorGps,
            "fmu/out/sensor_gps",
            self.gps_callback,
            qos_profile=qos_profile_sensor_data,
        )
        self._magnetometer_sub: rclpy.node.Subscription = self.create_subscription(
            VehicleMagnetometer,
            "fmu/out/vehicle_magnetometer",
            self.magnetometer_callback,
            qos_profile=qos_profile_sensor_data,
        )
        self._matched_points_sub: rclpy.node.Subscription = self.create_subscription(
            MatchedPoints,
            "camera/matched_points",
            self.matched_points_callback,
            qos_profile=qos_profile_sensor_data,
        )
        self._camera_info_sub: rclpy.node.Subscription = self.create_subscription(
            CameraInfo,
            "camera/camera_info",
            self.camera_info_callback,
            qos_profile=qos_profile_sensor_data,
        )

        parameters = gtsam.ISAM2Params()
        parameters.setRelinearizeThreshold(0.01)
        parameters.relinearizeSkip = 1
        self.isam: gtsam.ISAM2 = gtsam.ISAM2(parameters)

        init_graph = gtsam.NonlinearFactorGraph()
        init_values = gtsam.Values()
        init_graph, init_values = self.set_priors(graph=init_graph, values=init_values)
        init_graph, init_values = self.setup_imu_factors(
            graph=init_graph, values=init_values
        )
        self.isam.update(init_graph, init_values)

    def init_reference(self, lat_0, lon_0, alt_0):
        self.ref_alt = alt_0
        self.ref_lat = np.radians(lat_0)
        self.ref_lon = np.radians(lon_0)
        self.ref_sin_lat = np.sin(self.ref_lat)
        self.ref_cos_lat = np.cos(self.ref_lat)

    def project(self, lat, lon):
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        sin_lat = np.sin(lat_rad)
        cos_lat = np.cos(lat_rad)
        cos_d_lon = np.cos(lon_rad - self.ref_lon)
        arg = np.clip(
            self.ref_sin_lat * sin_lat + self.ref_cos_lat * cos_lat * cos_d_lon,
            -1.0,
            1.0,
        )
        c = np.arccos(arg)
        k = c / np.sin(c) if abs(c) > 0 else 1.0
        north = (
            k
            * (self.ref_cos_lat * sin_lat - self.ref_sin_lat * cos_lat * cos_d_lon)
            * 6371000
        )
        east = k * cos_lat * np.sin(lon_rad - self.ref_lon) * 6371000
        return north, east

    def add_gps_factor(
        self, graph: gtsam.NonlinearFactorGraph, key: int, msg: SensorGps
    ):
        if self.ref_sin_lat is None:
            self.init_reference(msg.latitude_deg, msg.longitude_deg, msg.altitude_msl_m)
        north, east = self.project(msg.latitude_deg, msg.longitude_deg)
        down = -(msg.altitude_msl_m - self.ref_alt)
        gps = gtsam.Point3(north, east, down)
        gpsNoise = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
        graph.add(gtsam.GPSFactor(key, gps, gpsNoise))

    def add_mag_factor(
        self, graph: gtsam.NonlinearFactorGraph, key: int, msg: VehicleMagnetometer
    ):
        self.magNoise = gtsam.noiseModel.Isotropic.Sigma(3, 0.5)
        yaw = np.arctan2(-msg.magnetometer_ga[1], msg.magnetometer_ga[0])
        rot = gtsam.Rot3.Yaw(yaw)
        pose = gtsam.Pose3(rot, gtsam.Point3(0, 0, 0))
        mag_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.1, 0.1, 0.1, 1e6, 1e6, 1e6])
        )
        graph.add(gtsam.PriorFactorPose3(key, pose, mag_noise))

    def set_priors(self, graph: gtsam.NonlinearFactorGraph, values: gtsam.Values):
        priorNoise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        )
        initial_pose = gtsam.Pose3(
            np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
        )
        graph.push_back(gtsam.PriorFactorPose3(X(0), initial_pose, priorNoise))
        values.insert(X(0), initial_pose)
        initial_velocity = gtsam.Point3(np.array([0.0, 0.0, 0.0]))
        velNoise = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
        graph.push_back(gtsam.PriorFactorVector(V(0), initial_velocity, velNoise))
        values.insert(V(0), initial_velocity)
        return graph, values

    def setup_imu_factors(
        self, graph: gtsam.NonlinearFactorGraph, values: gtsam.Values
    ):
        self.biasKey = B(0)
        self.biasNoise = gtsam.noiseModel.Isotropic.Sigma(6, 1e-3)
        graph.push_back(
            gtsam.PriorFactorConstantBias(
                self.biasKey, gtsam.imuBias.ConstantBias(), self.biasNoise
            )
        )
        values.insert(self.biasKey, gtsam.imuBias.ConstantBias())
        # define the preintegratedimumeasurements object here
        pim_params = self.preintegration_parameters()
        self.pim = gtsam.PreintegratedImuMeasurements(pim_params)
        return graph, values

    def preintegration_parameters(self):
        params = gtsam.PreintegrationParams.MakeSharedD(9.81)
        I = np.eye(3)  # noqa: E741
        params.setAccelerometerCovariance(I * 0.01)
        params.setGyroscopeCovariance(I * 0.001)
        params.setIntegrationCovariance(I * 0.01)
        params.setUse2ndOrderCoriolis(False)
        params.setOmegaCoriolis(np.array([0, 0, 0], dtype=float))
        return params

    def log_pose(self, pose: gtsam.Pose3):
        t = pose.translation()
        q = pose.rotation().toQuaternion()
        rr.set_time("keyframe", sequence=self.count)
        rr.log(
            "world/drone",
            rr.Transform3D(
                translation=[t[0], t[1], t[2]],
                rotation=rr.Quaternion(xyzw=[q.x(), q.y(), q.z(), q.w()]),
            ),
        )
        # accumulate and redraw the full path
        self.trajectory.append([t[0], t[1], t[2]])
        rr.log(
            "world/trajectory",
            rr.LineStrips3D([self.trajectory], colors=[[0, 200, 255]]),
        )

    def imu_callback(self, msg: SensorCombined):
        if self.prev_timestamp is None:
            self.prev_timestamp = msg.timestamp
            return

        dt_accel = msg.accelerometer_integral_dt * 1e-6

        gyro = np.array(msg.gyro_rad)
        accel = np.array(msg.accelerometer_m_s2)

        self.pim.integrateMeasurement(accel, gyro, dt_accel)

        if msg.timestamp - self.prev_timestamp > 50_000:  # 100hz
            graph = gtsam.NonlinearFactorGraph()
            values = gtsam.Values()

            i = self.count

            if self.latest_gps_msg is not None:
                self.add_gps_factor(graph, X(i + 1), self.latest_gps_msg)
                self.latest_gps_msg = None

            if self.latest_mag_msg is not None:
                self.add_mag_factor(graph, X(i + 1), self.latest_mag_msg)
                self.latest_mag_msg = None

            # grab current estimate
            pose = self.isam.calculateEstimatePose3(X(i))
            vel = self.isam.calculateEstimateVector(V(i))
            bias = self.isam.calculateEstimateConstantBias(self.biasKey)
            prev_state = gtsam.NavState(pose, vel)
            pred_state = self.pim.predict(prev_state, bias)
            values.insert(X(i + 1), pred_state.pose())
            values.insert(V(i + 1), pred_state.velocity())

            # create imu factor
            factor = gtsam.ImuFactor(
                X(i), V(i), X(i + 1), V(i + 1), self.biasKey, self.pim
            )
            graph.add(factor)

            # add bias factor periodically
            if self.count % 5 == 0 and self.count != 0:
                graph.add(
                    gtsam.BetweenFactorConstantBias(
                        self.biasKey,
                        self.biasKey + 1,
                        gtsam.imuBias.ConstantBias(),
                        self.biasNoise,
                    )
                )
                self.biasKey += 1
                values.insert(self.biasKey, gtsam.imuBias.ConstantBias())

            # optimize
            self.isam.update(graph, values)
            pose = self.isam.calculateEstimatePose3(X(i + 1))
            self.log_pose(pose)
            if self.count % 10 == 0:
                self.get_logger().info(str(pose))

            self.pim.resetIntegration()
            self.prev_timestamp = msg.timestamp
            self.count += 1

    def gps_callback(self, msg: SensorGps):
        self.latest_gps_msg = msg

    def magnetometer_callback(self, msg: VehicleMagnetometer):
        self.latest_mag_msg = msg

    def matched_points_callback(self, msg: Image): ...

    def camera_info_callback(self, msg: CameraInfo):
        if self.K is None:
            self.K = gtsam.Cal3_S2(
                msg.k[0],  # fx
                msg.k[4],  # fy
                msg.k[1],  # s (skew)
                msg.k[2],  # u0 (cx)
                msg.k[5],  # v0 (cy)
            )


def main(args=None):
    rclpy.init(args=args)

    px4_slam = PX4Slam()
    rclpy.spin(px4_slam)

    px4_slam.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
