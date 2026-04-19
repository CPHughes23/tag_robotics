import cv2
import numpy as np
import json
import math
import os
import argparse
from dotenv import load_dotenv

load_dotenv()

CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", 0))

CONFIG_FILE = "./robot/camera_config.json"

def load_config():
    if not os.path.exists(CONFIG_FILE):
        raise FileNotFoundError(
            f"'{CONFIG_FILE}' not found. Run hue_tuner.py and bbox_calibrate.py first."
        )
    with open(CONFIG_FILE, "r") as f:
        return json.load(f)

# Load at startup
config = load_config()

BLUE_LOWER  = np.array(config["blue"]["lower"])
BLUE_UPPER  = np.array(config["blue"]["upper"])
GREEN_LOWER = np.array(config["green"]["lower"])
GREEN_UPPER = np.array(config["green"]["upper"])

px_per_car  = config["car_scale"].get("pixels_per_car_length")  # None if not calibrated yet
MIN_BLOB_AREA = 200 # Minimum pixel area to count as a valid detection (filters noise)

# ── Helpers ───────────────────────────────────────────────────────────────────
def find_color_centroid(hsv_frame, lower, upper):
    """
    Masks the frame for a given HSV color range and returns the
    centroid of the largest blob found, or None if nothing detected.
    """
    mask = cv2.inRange(hsv_frame, lower, upper)

    # Morphological ops to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # removes small noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel) # fills small gaps

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, mask

    # Pick the largest contour
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_BLOB_AREA:
        return None, mask

    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None, mask

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy), mask

def px_to_car_lengths(px):
    """Convert a pixel distance to car-length units. Returns None if not calibrated."""
    if px_per_car:
        return px / px_per_car
    return None


def draw_tracking_overlay(frame, blue_center, green_center):
    if blue_center:
        cv2.circle(frame, blue_center, 10, (255, 100, 0), -1)
        cv2.putText(frame, "B", (blue_center[0] + 12, blue_center[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)

    if green_center:
        cv2.circle(frame, green_center, 10, (0, 255, 0), -1)
        cv2.putText(frame, "G", (green_center[0] + 12, green_center[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if blue_center and green_center:
        # Midpoint = position of the car
        mid_x = (blue_center[0] + green_center[0]) // 2
        mid_y = (blue_center[1] + green_center[1]) // 2
        midpoint = (mid_x, mid_y)

        cv2.circle(frame, midpoint, 6, (0, 255, 255), -1)
        cv2.line(frame, blue_center, green_center, (255, 255, 255), 2)

        # Draw orientation arrow from midpoint in the direction of the heading
        angle = math.atan2(green_center[1] - blue_center[1],
                           green_center[0] - blue_center[0])
        arrow_len = 60
        arrow_end = (
            int(mid_x + arrow_len * math.cos(angle)),
            int(mid_y + arrow_len * math.sin(angle))
        )
        cv2.arrowedLine(frame, midpoint, arrow_end, (0, 255, 255), 2, tipLength=0.3)

        angle_deg = math.degrees(angle)
        cv2.putText(frame, f"Heading: {angle_deg:.1f} deg",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Show scale warning if not calibrated
        if px_per_car is None:
            cv2.putText(frame, "Scale not calibrated - run bbox_calibrate.py",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 2)

    return frame


def main(show_mask=False):
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    last_blue, last_green = None, None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        blue_center,  blue_mask  = find_color_centroid(hsv, BLUE_LOWER,  BLUE_UPPER)
        green_center, green_mask = find_color_centroid(hsv, GREEN_LOWER, GREEN_UPPER)

        if blue_center:  last_blue  = blue_center
        if green_center: last_green = green_center

        frame = draw_tracking_overlay(frame, blue_center, green_center)

        if last_blue and last_green:
            angle = math.atan2(last_green[1] - last_blue[1],
                               last_green[0] - last_blue[0])

            # Distance in pixels and in car-lengths
            dx = last_green[0] - last_blue[0]
            dy = last_green[1] - last_blue[1]
            dist_px = math.sqrt(dx**2 + dy**2)
            dist_cl = px_to_car_lengths(dist_px)

            if dist_cl is not None:
                print(f"Pos: ({last_green[0]:.1f}, {last_green[1]:.1f})  "
                      f"Heading: {math.degrees(angle):.1f} deg  "
                      f"Dot sep: {dist_px:.1f}px = {dist_cl:.2f} car lengths")
            else:
                print(f"Pos: ({last_green[0]:.1f}, {last_green[1]:.1f})  "
                      f"Heading: {math.degrees(angle):.1f} deg  "
                      f"Dot sep: {dist_px:.1f}px (not calibrated)")

        if show_mask:
            debug = cv2.hconcat([
                cv2.cvtColor(blue_mask,  cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(green_mask, cv2.COLOR_GRAY2BGR)
            ])
            cv2.imshow("Masks: Blue | Green", debug)

        cv2.imshow("Color Tracking", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--show-mask", action="store_true")
    args = parser.parse_args()
    main(show_mask=args.show_mask)