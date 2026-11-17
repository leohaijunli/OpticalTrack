import cv2 as cv
import numpy as np
from LowPassFilter import LowPassFilter
import time

class KalmanFilter:
    def __init__(self, num_objects, process_noise=1e-2, measurement_noise=1.0,
                 reacquire_gap_s=0.5, sampling_frequency=60.0, max_speed=10.0,
                 reacquire_inflation=5.0):
        state_dim = 9
        measurement_dim = 6
        dt = 0.1
        self.max_speed = float(max_speed)
        if not np.isfinite(self.max_speed) or self.max_speed <= 0:
            raise ValueError("max_speed must be positive")
        if not np.isfinite(reacquire_inflation) or reacquire_inflation < 0:
            raise ValueError("reacquire_inflation must be non-negative")
        self.reacquire_inflation = float(reacquire_inflation)
        self.sampling_frequency = float(sampling_frequency)
        self.kalmans = []
        self.prev_measurement_time = 0
        self.prev_positions = []

        self.low_pass_filter_xy = []
        self.low_pass_filter_z = []
        self.heading_low_pass_filter = []
        self.num_objects = num_objects
        self.last_seen = [None] * num_objects
        self.reacquire_gap_s = reacquire_gap_s
        if not np.isfinite(sampling_frequency) or sampling_frequency <= 0:
            raise ValueError("sampling_frequency must be positive")
        # Same normalized Butterworth bandwidth as the original 20 Hz/60 Hz
        # setup, expressed at the actual replay/capture rate.
        cutoff_hz = sampling_frequency / 3.0

        for i in range(num_objects):
            self.prev_positions.append([0,0,0])
            self.kalmans.append(cv.KalmanFilter(state_dim, measurement_dim))
            self.kalmans[i].transitionMatrix = np.array([[1, 0, 0, dt, 0, 0, 0.5*dt**2, 0, 0],
                                                    [0, 1, 0, 0, dt, 0, 0, 0.5*dt**2, 0],
                                                    [0, 0, 1, 0, 0, dt, 0, 0, 0.5*dt**2],
                                                    [0, 0, 0, 1, 0, 0, dt, 0, 0],
                                                    [0, 0, 0, 0, 1, 0, 0, dt, 0],
                                                    [0, 0, 0, 0, 0, 1, 0, 0, dt],
                                                    [0, 0, 0, 0, 0, 0, 1, 0, 0],
                                                    [0, 0, 0, 0, 0, 0, 0, 1, 0],
                                                    [0, 0, 0, 0, 0, 0, 0, 0, 1]], dtype=np.float32)

            self.kalmans[i].processNoiseCov = np.eye(state_dim, dtype=np.float32) * process_noise
            self.kalmans[i].measurementNoiseCov = np.eye(measurement_dim, dtype=np.float32) * measurement_noise
            self.kalmans[i].measurementMatrix = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                                                    [0, 1, 0, 0, 0, 0, 0, 0, 0],
                                                    [0, 0, 1, 0, 0, 0, 0, 0, 0],
                                                    [0, 0, 0, 1, 0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 1, 0, 0, 0]], dtype=np.float32)
            
            self.kalmans[i].statePost = np.zeros((9,1), dtype=np.float32)


            self.low_pass_filter_xy.append(LowPassFilter(cutoff_frequency=cutoff_hz, sampling_frequency=sampling_frequency, dims=2))
            self.low_pass_filter_z.append(LowPassFilter(cutoff_frequency=cutoff_hz, sampling_frequency=sampling_frequency, dims=1))
            self.heading_low_pass_filter.append(LowPassFilter(cutoff_frequency=cutoff_hz, sampling_frequency=sampling_frequency, dims=2))
    

    def predict_location(self, objects, timestamp=None):
        res = []

        now = time.time() if timestamp is None else float(timestamp)
        if not np.isfinite(now):
            raise ValueError("a finite measurement timestamp is required")
        if any(last is not None and now < last for last in self.last_seen):
            self.reset()
        self.prev_measurement_time = now

        for drone_index in range(0, self.num_objects):
            possible_new_objects = [object for object in objects if object["droneIndex"] == drone_index]
            possible_new_positions = [x["pos"] for x in possible_new_objects]

            if len(possible_new_objects) == 0:
                continue

            kalman = self.kalmans[drone_index]
            last = self.last_seen[drone_index]
            initialized = last is None or now - last > self.reacquire_gap_s
            if last is not None and now == last:
                continue  # never apply the same captured frame twice
            dt = 1e-3 if initialized else max(now - last, 1e-6)

            kalman.transitionMatrix[:3, 3:6] = dt * np.eye(3)
            kalman.transitionMatrix[3:6, 6:9] = dt * np.eye(3)
            kalman.transitionMatrix[:3, 6:9] = 0.5 * dt**2 * np.eye(3)

            if initialized:
                # Prime the previous measurement instead of inventing a
                # velocity from [0,0,0].  Per-body timestamps avoid inflated
                # velocity after a missed observation; predict/correct and
                # the upstream statePre output convention remain unchanged.
                A = np.zeros((9, 1), dtype=np.float32)
                A[0:3] = possible_new_positions[0].reshape((3, 1))
                kalman.statePost = A
                kalman.statePre = A
                kalman.errorCovPost = np.zeros((9, 9), dtype=np.float32)
                self.prev_positions[drone_index] = possible_new_positions[0].astype(np.float32).copy()
                self.low_pass_filter_xy[drone_index].buffered_data = np.empty((0, 2))
                self.low_pass_filter_z[drone_index].buffered_data = np.empty((0, 1))
                self.heading_low_pass_filter[drone_index].buffered_data = np.empty((0, 2))
            else:
                # A short dropout (below reacquire_gap_s) keeps the track, but
                # the covariance left over from the last update no longer
                # describes an estimate that was never measured for several
                # frames.  Without this, a reacquiring frame is mostly the
                # extrapolation: a body that decelerated during the gap keeps
                # the stale velocity and lands centimetres away even though the
                # fresh fix is millimetres from the truth.  Inflating in units
                # of the measurement noise reopens exactly the states that
                # went unobserved, and is scale-free in Q/R.
                missed = int(round((now - last) * self.sampling_frequency)) - 1
                if missed > 0 and self.reacquire_inflation:
                    grow = self.reacquire_inflation * missed
                    kalman.errorCovPost[:3, :3] += (
                        grow * kalman.measurementNoiseCov[:3, :3]).astype(np.float32)
                    kalman.errorCovPost[3:6, 3:6] += (
                        grow * kalman.measurementNoiseCov[3:6, 3:6]).astype(np.float32)

            
            predicted_location = kalman.predict()[:3].T[0]
            distances_to_predicted_location = np.sqrt(np.sum((possible_new_positions - predicted_location)**2, axis=1))
            closest_match_i = np.argmin(distances_to_predicted_location)
            new_pos = possible_new_positions[closest_match_i].astype(np.float32)
            new_vel = ((new_pos - self.prev_positions[drone_index]) / dt).astype(np.float32)
            # Configurable bound on the velocity MEASUREMENT magnitude, for
            # the selected position reference. This is not an innovation test
            # and does not establish that a position measurement is valid.
            if self.max_speed and not initialized and np.linalg.norm(new_vel) > self.max_speed:
                new_vel = new_vel / np.linalg.norm(new_vel) * self.max_speed
            self.prev_positions[drone_index] = new_pos
            self.last_seen[drone_index] = now

            # cv2 >= 5.0 requires the measurement as an explicit column vector
            # (OpenCV 4.x accepted a flat (6,) array).
            kalman.correct(np.concatenate((new_pos, new_vel)).reshape(6, 1))
            # The value reported for this capture is the posterior estimate.
            # OpenCV's ``correct`` also writes ``statePre`` in place (5.x), so
            # the historical read of ``statePre`` never actually returned the
            # a-priori prediction; read ``statePost`` so the intent is explicit
            # and version-independent.
            predicted_state = kalman.statePost[:, 0].copy()  # p, v, a at this capture

            heading = possible_new_objects[closest_match_i]["heading"]
            # The base axis is pi-periodic. Filtering folded angles directly
            # interpolates +89/-89 degrees through zero (the perpendicular
            # axis). Double-angle sine/cosine preserves the axial topology.
            heading_components = self.heading_low_pass_filter[drone_index].filter(
                np.array([np.cos(2 * heading), np.sin(2 * heading)]))
            if np.linalg.norm(heading_components) > 1e-12:
                heading = 0.5 * np.arctan2(heading_components[1], heading_components[0])
            # heading, self.z = signal.lfilter(self.b, 1, [heading], zi=self.z)

            vel = predicted_state[3:6].copy()
            vel[0:2] = self.low_pass_filter_xy[drone_index].filter(vel[0:2])
            vel[2] = self.low_pass_filter_z[drone_index].filter(vel[2])[0]

            entry = {
                "pos": predicted_state[:3],
                "vel": vel,
                "acc": predicted_state[6:9].copy(),
                "heading": heading,
                "droneIndex": drone_index
            }
            # Rigid-body pose and identity evidence are geometric, not part of
            # the upstream 9-state position filter.  Pass them through
            # unchanged so consumers see the accepted measurement's pose.
            measurement = possible_new_objects[closest_match_i]
            for key in ("pose_r", "pose_t", "pose_euler_zyx_deg", "pose_tilt_deg",
                         "pose_rmse_m", "pose_triangle_rmse_m", "pose_mode", "pose_ambiguous",
                         "pose_disambiguation", "markers_used",
                        "marker_ids", "extra_residuals_m", "reference_ambiguous",
                        "name", "timestamp", "error", "position_reference", "base_midpoint_measured"):
                if key in measurement:
                    entry[key] = measurement[key]
            res.append(entry)

        return res


    def reset(self):
        self.prev_measurement_time = 0
        self.last_seen = [None] * self.num_objects

        for i, kalman in enumerate(self.kalmans):
            kalman.statePost = np.zeros((9,1), dtype=np.float32)
            self.prev_positions[i] = np.array([0,0,0])
