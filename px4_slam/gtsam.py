import gtsam
import numpy as np
import rclpy
import rerun as rr
from gtsam.symbol_shorthand import B, G, V, X
from px4_msgs.msg import SensorCombined, VehicleGlobalPosition
from pyproj import Transformer
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data


class PX4Slam(Node):
    biasKey: int
    biasNoise: gtsam.noiseModel.Isotropic
    gpsKey: int
    gpsNoise: gtsam.noiseModel.Isotropic
    pim: gtsam.PreintegratedImuMeasurements
    latest_gps_msg: VehicleGlobalPosition | None
    origin: np.ndarray | None

    def __init__(self):
        super().__init__("px4_slam")

        rr.init("rerun_gtsam")
        rr.spawn()

        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        rr.log("world/drone", rr.TransformAxes3D(axis_length=1.0), static=True)
        self.trajectory: list[list[float]] = []

        self.origin = None

        self.transformer = Transformer.from_crs(
            "epsg:4326", "epsg:4978", always_xy=True
        )

        self._imu_sub: rclpy.node.Subscription = self.create_subscription(
            SensorCombined,
            "/fmu/out/sensor_combined",
            self.imu_callback,
            qos_profile=qos_profile_sensor_data,
        )
        self._global_position_sub: rclpy.node.Subscription = self.create_subscription(
            VehicleGlobalPosition,
            "/fmu/out/vehicle_global_position",
            self.global_position_callback,
            qos_profile=qos_profile_sensor_data,
        )

        self.latest_gps_msg = None
        self.prev_timestamp = None
        self.count = 0

        self.isam: gtsam.ISAM2 = gtsam.ISAM2(gtsam.ISAM2Params())
        init_graph = gtsam.NonlinearFactorGraph()
        init_values = gtsam.Values()

        init_graph, init_values = self.set_priors(graph=init_graph, values=init_values)

        init_graph, init_values = self.setup_imu_factors(
            graph=init_graph, values=init_values
        )
        self.gpsKey = G(0)

        self.isam.update(init_graph, init_values)

    def add_gps_factor(
        self, graph: gtsam.NonlinearFactorGraph, key: int, msg: VehicleGlobalPosition
    ):
        x, y, z = self.transformer.transform(msg.lon, msg.lat, msg.alt)
        ecef = np.array([x, y, z])

        if self.origin is None:
            self.origin = ecef

        local = ecef - self.origin  # meters, relative to first fix

        self.get_logger().info(f"local: {local}")

        self.gpsNoise = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
        gps = gtsam.Point3(local[0], local[1], local[2])
        graph.add(gtsam.GPSFactor(key, gps, self.gpsNoise))

    def set_priors(self, graph: gtsam.NonlinearFactorGraph, values: gtsam.Values):
        priorNoise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.1, 0.1, 0.1, 0.3, 0.3, 0.3])
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
        self.biasNoise = gtsam.noiseModel.Isotropic.Sigma(6, 0.1)
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
        params.setAccelerometerCovariance(I * 0.1)
        params.setGyroscopeCovariance(I * 0.1)
        params.setIntegrationCovariance(I * 0.1)
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

        if msg.timestamp - self.prev_timestamp > 100_000:  # 10hz
            graph = gtsam.NonlinearFactorGraph()
            values = gtsam.Values()

            i = self.count

            # create imu factor
            factor = gtsam.ImuFactor(
                X(i), V(i), X(i + 1), V(i + 1), self.biasKey, self.pim
            )
            graph.add(factor)

            if self.latest_gps_msg is not None:
                self.add_gps_factor(graph, X(i + 1), self.latest_gps_msg)
                self.gpsKey += 1
                self.latest_gps_msg = None

            # grab current estimate
            pose = self.isam.calculateEstimatePose3(X(i))
            vel = self.isam.calculateEstimateVector(V(i))
            bias = self.isam.calculateEstimateConstantBias(self.biasKey)
            prev_state = gtsam.NavState(pose, vel)
            pred_state = self.pim.predict(prev_state, bias)
            values.insert(X(i + 1), pred_state.pose())
            values.insert(V(i + 1), pred_state.velocity())

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

            # self.log_pose(result.atPose3(X(i))
            self.pim.resetIntegration()
            self.prev_timestamp = msg.timestamp
            self.count += 1

            # result = self.isam.calculateEstimate()
            # result.print()
            # self.isam.getFactorsUnsafe().print()

    def global_position_callback(self, msg: VehicleGlobalPosition):
        self.latest_gps_msg = msg


def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = PX4Slam()
    rclpy.spin(minimal_subscriber)

    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
