"""Regression checks for the evidence used to assess edge deployment."""
import hashlib
import json
import unittest
from unittest import mock

import cv2 as cv
import numpy as np

from tools.edge_equivalence_check import (assemble_frame, bandwidth_summary,
                                          edge_video_packets, json_bytes, parse_frames,
                                          validate_capture)


class EdgeEquivalenceTests(unittest.TestCase):
    def test_worker_sparse_frames_contain_the_requested_pixels(self):
        capture = mock.Mock()
        cursor = [0]
        capture.get.side_effect = {cv.CAP_PROP_FPS: 30, cv.CAP_PROP_FRAME_COUNT: 600,
                                   cv.CAP_PROP_FRAME_WIDTH: 96, cv.CAP_PROP_FRAME_HEIGHT: 80}.get

        def image(frame):
            pixels = np.zeros((80, 96, 3), dtype=np.uint8)
            cv.circle(pixels, (20 + frame % 60, 40), 3, (255, 255, 255), -1)
            return pixels

        def seek(prop, frame):
            self.assertEqual(prop, cv.CAP_PROP_POS_FRAMES)
            cursor[0] = frame
            return True

        def read():
            pixels = image(cursor[0])
            cursor[0] += 1
            return True, pixels

        capture.set.side_effect = seek
        capture.read.side_effect = read
        frames = [0, 50, 599]
        job = {"path": "mock.mp4", "calibration": "mock.json", "camera_index": 0,
               "camera": "CAM01", "frames": frames, "fps": 30,
               "detector": {"blur_size": 1, "threshold": 140, "min_area": 3, "max_area": 200},
               "session_id": "test", "calibration_sha256": "cal", "detector_sha256": "det"}
        param = {"intrinsic_matrix": np.eye(3), "distortion_coef": np.zeros(5)}
        with mock.patch("tools.edge_equivalence_check.cv.VideoCapture", return_value=capture), \
                mock.patch("tools.edge_equivalence_check.cv.setNumThreads"), \
                mock.patch("tools.edge_equivalence_check.load_calibration", return_value=({}, [param], [])):
            result = edge_video_packets(job)
        packets = [json.loads(raw) for raw in result["packets"]]
        self.assertEqual([packet["frame_id"] for packet in packets], frames)
        self.assertEqual([packet["points"] for packet in packets], [[[20 + f % 60, 40]] for f in frames])
        self.assertEqual(result["pixel_hashes"], [hashlib.sha256(image(f)).hexdigest() for f in frames])
        capture.release.assert_called_once()

    def packets(self):
        expected = {"session_id": "test", "calibration_sha256": "cal", "detector_sha256": "det"}
        common = dict(expected, schema=1, frame_id=8, exposure_midpoint_ns=round(8e9 / 30),
                      valid=True, clock_domain="simulation", coordinate_space="undistorted_pixels")
        packets = [json_bytes(dict(common, camera="CAM01", points=[[80, 4], [12, 9]])),
                   json_bytes(dict(common, camera="CAM02", points=[]))]
        return expected, packets

    def test_delivery_order_does_not_change_camera_or_point_order(self):
        expected, packets = self.packets()
        result = assemble_frame(packets[::-1], ["CAM01", "CAM02"], 8, 30, expected)
        self.assertEqual(result, [[[80, 4], [12, 9]], []])

    def test_packet_roundtrip_does_not_quantize_subpixel_coordinates(self):
        expected, packets = self.packets()
        packet = json.loads(packets[0])
        packet["points"] = [[80.25, 4.125], [12.333333333333334, 9.75]]
        packets[0] = json_bytes(packet)
        result = assemble_frame(packets[::-1], ["CAM01", "CAM02"], 8, 30, expected)
        np.testing.assert_array_equal(result[0], packet["points"])

    def test_duplicate_missing_or_mismatched_packets_fail(self):
        expected, packets = self.packets()
        for bad in ([packets[0]], [packets[0], packets[0]]):
            with self.subTest(packets=bad), self.assertRaises(ValueError):
                assemble_frame(bad, ["CAM01", "CAM02"], 8, 30, expected)
        with self.assertRaises(ValueError):
            assemble_frame(packets, ["CAM01", "CAM02"], 9, 30, expected)
        with self.assertRaises(ValueError):
            assemble_frame(packets, ["CAM01", "CAM02"], 8, 30, dict(expected, calibration_sha256="wrong"))

    def test_bandwidth_uses_bits_and_counts_the_total_only_once(self):
        result = bandwidth_summary([[100, 100]] * 6, [[800, 1280]] * 6, 2, 60, 6)
        self.assertEqual(result["point_payload_kbps_per_camera"], [48] * 6)
        self.assertAlmostEqual(result["point_payload_total_mbps"], 0.288)
        self.assertAlmostEqual(result["point_wire_total_mbps"], 0.47808)
        self.assertAlmostEqual(result["nominal_1gbe_remaining_mbps"], 963.52192)
        self.assertAlmostEqual(result["raw_gray8_total_mbps"], 2949.12)

    def test_sixty_fps_budget_cannot_relabel_thirty_fps_source(self):
        capture = mock.Mock()
        capture.get.return_value = 30
        with self.assertRaisesRegex(ValueError, "source FPS mismatch"):
            validate_capture(capture, "test.mp4", [0], 60)

    def test_frame_selection_rejects_empty_reverse_and_repeated_sequences(self):
        self.assertEqual(parse_frames("0-599"), list(range(600)))
        self.assertEqual(parse_frames("0,50,599"), [0, 50, 599])
        for text in ("", "10-0", "3,3", "4,2", "-1"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_frames(text)


if __name__ == "__main__":
    unittest.main()
