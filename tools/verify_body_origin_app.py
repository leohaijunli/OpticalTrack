"""Exercise the running application's real MJPEG + Socket.IO tracking path."""
import argparse
import json
from pathlib import Path
import queue
import threading
import time

import numpy as np
import requests
import socketio

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:3001")
    parser.add_argument("--through-time", type=float, default=18.3)
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--out", type=Path, default=ROOT / "out/rig3/app_body_origin_check.json")
    args = parser.parse_args()
    response = requests.get(args.url + "/api/tracking-config", timeout=10)
    response.raise_for_status()
    config = response.json()
    if config["position_reference"] != "body_origin" or len(config["kalman_state"]) != 9:
        raise RuntimeError("application has not loaded the body-origin / nine-state configuration")

    packets = queue.Queue()
    client = socketio.Client(reconnection=False)
    client.on("object-points", lambda payload: packets.put(payload))
    stream = None
    stopped = threading.Event()
    reader = None
    byte_count = [0]
    try:
        client.connect(args.url, transports=["polling"], wait_timeout=10)
        stream = requests.get(args.url + "/api/camera-stream", stream=True, timeout=(10, 30))
        stream.raise_for_status()
        if "multipart/x-mixed-replace" not in stream.headers.get("Content-Type", ""):
            raise RuntimeError("camera stream is not MJPEG")

        def consume():
            try:
                for chunk in stream.iter_content(65536):
                    byte_count[0] += len(chunk)
                    if stopped.is_set():
                        break
            except Exception as error:
                if not stopped.is_set():
                    packets.put(error)

        reader = threading.Thread(target=consume, daemon=True)
        reader.start()
        client.call("triangulate-points", {"startOrStop": "start", "cameraPoses": config["camera_poses"],
                                          "toWorldCoordsMatrix": None}, timeout=15)
        client.call("locate-objects", {"startOrStop": "start"}, timeout=15)
        deadline = time.monotonic() + args.timeout
        captures, prior_fixes, per_body = set(), [], {}
        first, last = None, None
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("application did not reach the requested capture time")
            payload = packets.get(timeout=remaining)
            if isinstance(payload, Exception):
                raise payload
            tracking = payload.get("tracking")
            if tracking is None:
                continue
            stamp = tracking["timestamp"]
            if tracking["position_reference"] != "body_origin":
                raise AssertionError("mixed position references in the live payload")
            first = stamp if first is None else first
            last = stamp
            captures.add(stamp)
            for obj in payload["objects"]:
                np.testing.assert_allclose(obj["pos"], obj["pose_t"], atol=1e-9)
                if obj["pose_disambiguation"] == "temporal_prior":
                    prior_fixes.append({"t": stamp, "body_id": obj["droneIndex"]})
            for obj in payload["filtered_objects"]:
                if obj["position_reference"] != "body_origin":
                    raise AssertionError("Kalman output used the wrong reference")
                for key in ("pos", "vel", "acc"):
                    value = np.asarray(obj[key])
                    if value.shape != (3,) or not np.all(np.isfinite(value)):
                        raise AssertionError(f"invalid {key} in live output")
                bid = str(obj["droneIndex"])
                per_body[bid] = per_body.get(bid, 0) + 1
            if last >= args.through_time and prior_fixes:
                break
        report = {"position_reference": "body_origin", "kalman_state": config["kalman_state"],
                  "camera_count": len(config["camera_poses"]), "captures": len(captures),
                  "first_t": first, "last_t": last, "body_observations": per_body,
                  "temporal_prior_fixes": prior_fixes, "mjpeg_bytes": byte_count[0],
                  "checks": "raw pos equals fitted origin; finite p/v/a; temporal prior observed over the live API"}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report), flush=True)
    finally:
        stopped.set()
        if stream is not None:
            stream.close()
        if reader is not None:
            reader.join(timeout=5)
        if client.connected:
            client.disconnect()


if __name__ == "__main__":
    main()
