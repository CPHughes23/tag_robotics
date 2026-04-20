"""
rc_auto.py  –  Autonomous RC car controller
Click a target on the camera feed, and the trained policy drives the car there.

Usage:
    python rc_auto.py [--model MODEL.pt] [--show-mask]

Requirements: same camera_config.json as tracker.py
Serial port / baud read from .env  (SERIAL_PORT, SERIAL_BAUD)
"""

import argparse
import json
import math
import os
import time

import cv2
import numpy as np
import torch
import serial
from dotenv import load_dotenv

load_dotenv()

# Config 
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", 0))
SERIAL_PORT  = os.getenv("SERIAL_PORT", "COM3")
SERIAL_BAUD  = int(os.getenv("SERIAL_BAUD", 9600))
CONFIG_FILE  = "./robot/camera_config.json"
DEFAULT_MODEL= "./robot/policy.pt"

MIN_BLOB_AREA    = 200
REACH_THRESHOLD  = 1.5   # car lengths: stop sending commands inside this radius

# Load camera config
if not os.path.exists(CONFIG_FILE):
    raise FileNotFoundError(f"'{CONFIG_FILE}' not found. Run hue_tuner.py / bbox_calibrate.py first.")

with open(CONFIG_FILE) as f:
    config = json.load(f)

BLUE_LOWER  = np.array(config["blue"]["lower"])
BLUE_UPPER  = np.array(config["blue"]["upper"])
GREEN_LOWER = np.array(config["green"]["lower"])
GREEN_UPPER = np.array(config["green"]["upper"])
px_per_car  = config["car_scale"].get("pixels_per_car_length")  # may be None

# Serial helpers
def open_serial():
    try:
        ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=0.1)
        time.sleep(2) # let Arduino boot
        print(f"Serial open on {SERIAL_PORT} @ {SERIAL_BAUD}")
        return ser
    except serial.SerialException as e:
        print(f"[WARN] Could not open serial port: {e}")
        print("       Running in DRY-RUN mode (no car commands sent).")
        return None


def send_action(ser, drive: float, steer: float):
    if ser is None:
        return

    # Forward / backward axis
    if drive < -0.5:
        ser.write(b'0')   # forward
    elif drive > 0.5:
        ser.write(b'1')   # reverse
    else:
        ser.write(b'f')   # stop drive axis

    # Steering axis
    if steer < -0.25:
        ser.write(b'2')   # left
    elif steer > 0.25:
        ser.write(b'3')   # right
    else:
        ser.write(b'r')   # straight


def stop_car(ser):
    if ser:
        ser.write(b'4')   # full stop

# Vision helpers
def find_color_centroid(hsv_frame, lower, upper):
    mask = cv2.inRange(hsv_frame, lower, upper)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,   kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_BLOB_AREA:
        return None, mask

    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None, mask

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy), mask

# Observation builder
def build_obs(blue_center, green_center, target_px):
    if blue_center is None or green_center is None or target_px is None:
        return None

    # Heading
    # green dot is the "forward" end (same convention as draw_tracking_overlay)
    dx_heading = green_center[0] - blue_center[0]
    dy_heading = green_center[1] - blue_center[1]
    heading = math.atan2(dy_heading, dx_heading)   # radians, camera frame

    # Car midpoint
    mid_x = (blue_center[0] + green_center[0]) / 2.0
    mid_y = (blue_center[1] + green_center[1]) / 2.0

    # World-frame vector to target
    to_tx = target_px[0] - mid_x
    to_ty = target_px[1] - mid_y

    dist_px = math.sqrt(to_tx**2 + to_ty**2)

    # Car-length normalization
    car_length_px = px_per_car if px_per_car else 1.0   # fallback: pixels

    # Local frame (rotate into car frame)
    sin_h = math.sin(heading)
    cos_h = math.cos(heading)
    local_x =  cos_h * to_tx + sin_h * to_ty
    local_y = -sin_h * to_tx + cos_h * to_ty

    # Normalize by car length
    local_x /= car_length_px
    local_y /= car_length_px
    distance  = dist_px / car_length_px

    obs = torch.tensor([[heading, local_x, local_y, distance]], dtype=torch.float32)
    return obs, distance   # also return scalar distance for threshold check

# Policy loader

OBS_DIM    = 4   # [heading, local_x, local_y, distance]
ACTION_DIM = 2   # [drive, steer]

def build_mlp(obs_dim, hidden_dims, action_dim, activation=torch.nn.ELU):
    layers = []
    in_dim = obs_dim
    for h in hidden_dims:
        layers.append(torch.nn.Linear(in_dim, h))
        layers.append(activation())
        in_dim = h
    # std is stored separately in distribution.std_param; mlp output is just the means
    layers.append(torch.nn.Linear(in_dim, action_dim))
    return torch.nn.Sequential(*layers)


def load_policy(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    # rsl_rl saves actor and critic separately
    if "actor_state_dict" in checkpoint:
        state = checkpoint["actor_state_dict"]
        strip_prefix = False
    elif "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
        strip_prefix = True
    elif "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
        strip_prefix = True
    else:
        raise KeyError(f"Unexpected checkpoint keys: {list(checkpoint.keys())}")

    print("Keys found in checkpoint state dict:")
    for k, v in state.items():
        print(f"  {k}: {v.shape}")

    actor = build_mlp(OBS_DIM, [128, 64, 32], ACTION_DIM)

    if strip_prefix:
        actor_state = {k[len("actor."):]: v for k, v in state.items() if k.startswith("actor.")}
    else:
        actor_state = {k[len("mlp."):]: v for k, v in state.items() if k.startswith("mlp.")}

    actor.load_state_dict(actor_state, strict=True)
    actor.eval()
    print(f"Policy loaded from '{path}'")
    return actor


def infer(policy, obs: torch.Tensor):
    with torch.no_grad():
        out = policy(obs)   # shape (1, 2) — means only
    mean_drive = float(out[0, 0])
    mean_steer = float(out[0, 1])

    # Discretize as _pre_physics_step does in training
    drive = 1.0 if mean_drive > 0.33 else (-1.0 if mean_drive < -0.33 else 0.0)
    steer = 0.5 if mean_steer > 0.17 else (-0.5 if mean_steer < -0.17 else 0.0)
    return drive, steer

# Mouse callback
class TargetSelector:
    def __init__(self):
        self.target = None

    def callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.target = (x, y)
            print(f"New target set: ({x}, {y})")

# Overlay drawing
def draw_overlay(frame, blue_center, green_center, target, reached):
    # Dots
    if blue_center:
        cv2.circle(frame, blue_center, 10, (255, 100, 0), -1)
        cv2.putText(frame, "B", (blue_center[0]+12, blue_center[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
    if green_center:
        cv2.circle(frame, green_center, 10, (0, 255, 0), -1)
        cv2.putText(frame, "G", (green_center[0]+12, green_center[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if blue_center and green_center:
        mid_x = (blue_center[0] + green_center[0]) // 2
        mid_y = (blue_center[1] + green_center[1]) // 2
        midpoint = (mid_x, mid_y)

        cv2.circle(frame, midpoint, 6, (0, 255, 255), -1)
        cv2.line(frame, blue_center, green_center, (255, 255, 255), 2)

        angle = math.atan2(green_center[1] - blue_center[1],
                           green_center[0] - blue_center[0])
        arrow_end = (
            int(mid_x + 60 * math.cos(angle)),
            int(mid_y + 60 * math.sin(angle))
        )
        cv2.arrowedLine(frame, midpoint, arrow_end, (0, 255, 255), 2, tipLength=0.3)
        cv2.putText(frame, f"Heading: {math.degrees(angle):.1f} deg",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Target
    if target:
        color = (0, 200, 0) if reached else (0, 0, 255)
        cv2.drawMarker(frame, target, color, cv2.MARKER_CROSS, 30, 3)
        label = "TARGET (reached)" if reached else "TARGET"
        cv2.putText(frame, label, (target[0]+15, target[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Instructions
    cv2.putText(frame, "Click = set target | ESC = quit",
                (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (200, 200, 200), 1)

    if px_per_car is None:
        cv2.putText(frame, "Scale not calibrated – distances in pixels",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 100, 255), 1)

    return frame

# Main loop
def main(model_path, show_mask=False):
    policy = load_policy(model_path)
    ser = open_serial()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    selector = TargetSelector()
    win_name = "RC Auto – click to set target"
    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, selector.callback)

    last_blue = last_green = None
    reached = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            blue_center,  blue_mask  = find_color_centroid(hsv, BLUE_LOWER,  BLUE_UPPER)
            green_center, green_mask = find_color_centroid(hsv, GREEN_LOWER, GREEN_UPPER)

            if blue_center:  last_blue  = blue_center
            if green_center: last_green = green_center

            target = selector.target

            # Control
            if last_blue and last_green and target:
                result = build_obs(last_blue, last_green, target)

                if result is not None:
                    obs, distance_cl = result

                    if distance_cl < REACH_THRESHOLD:
                        # At target – stop the car
                        if not reached:
                            stop_car(ser)
                            print(f"Reached target! ({distance_cl:.2f} car lengths)")
                        reached = True
                    else:
                        reached = False
                        drive, steer = infer(policy, obs)
                        send_action(ser, drive, steer)
                        print(f"dist={distance_cl:.2f}cl  drive={drive:+.1f}  steer={steer:+.2f}")
            else:
                # No valid tracking, stop for safety
                stop_car(ser)

            frame = draw_overlay(frame, last_blue, last_green, target, reached)

            if show_mask:
                debug = cv2.hconcat([
                    cv2.cvtColor(blue_mask,  cv2.COLOR_GRAY2BGR),
                    cv2.cvtColor(green_mask, cv2.COLOR_GRAY2BGR),
                ])
                cv2.imshow("Masks: Blue | Green", debug)

            cv2.imshow(win_name, frame)
            if cv2.waitKey(1) & 0xFF == 27:   # ESC
                break

    finally:
        stop_car(ser)
        if ser:
            ser.close()
        cap.release()
        cv2.destroyAllWindows()
        print("Stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     default=DEFAULT_MODEL, help="Path to TorchScript .pt policy")
    parser.add_argument("--show-mask", action="store_true",   help="Show HSV debug masks")
    args = parser.parse_args()
    main(args.model, args.show_mask)