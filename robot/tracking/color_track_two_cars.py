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

BLUE_LOWER   = np.array(config["blue"]["lower"])
BLUE_UPPER   = np.array(config["blue"]["upper"])
GREEN_LOWER  = np.array(config["green"]["lower"])
GREEN_UPPER  = np.array(config["green"]["upper"])
YELLOW_LOWER = np.array(config["yellow"]["lower"])
YELLOW_UPPER = np.array(config["yellow"]["upper"])
PINK_LOWER   = np.array(config["pink"]["lower"])
PINK_UPPER   = np.array(config["pink"]["upper"])

px_per_car  = config["car_scale"].get("pixels_per_car_length")  # None if not calibrated yet
MIN_BLOB_AREA = 200  # Minimum pixel area to count as a valid detection

# Helpers
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


def draw_car_overlay(frame, front_center, back_center, label, dot_color_front, dot_color_back, text_y_offset=0):
    """
    Draws tracking overlay for a single car.
    front_center and back_center are the two color dot positions.
    The heading arrow points from back -> front.
    """
    if front_center:
        cv2.circle(frame, front_center, 10, dot_color_front, -1)
        cv2.putText(frame, label[0],  # first letter as front marker
                    (front_center[0] + 12, front_center[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, dot_color_front, 2)

    if back_center:
        cv2.circle(frame, back_center, 10, dot_color_back, -1)
        cv2.putText(frame, label[1],  # second letter as back marker
                    (back_center[0] + 12, back_center[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, dot_color_back, 2)

    if front_center and back_center:
        mid_x = (front_center[0] + back_center[0]) // 2
        mid_y = (front_center[1] + back_center[1]) // 2
        midpoint = (mid_x, mid_y)

        cv2.circle(frame, midpoint, 6, (0, 255, 255), -1)
        cv2.line(frame, front_center, back_center, (255, 255, 255), 2)

        # Heading from back dot toward front dot
        angle = math.atan2(front_center[1] - back_center[1],
                           front_center[0] - back_center[0])
        arrow_len = 60
        arrow_end = (
            int(mid_x + arrow_len * math.cos(angle)),
            int(mid_y + arrow_len * math.sin(angle))
        )
        cv2.arrowedLine(frame, midpoint, arrow_end, (0, 255, 255), 2, tipLength=0.3)

        angle_deg = math.degrees(angle)
        cv2.putText(frame, f"{label[2:]}: Heading {angle_deg:.1f} deg",
                    (10, 30 + text_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return frame


def print_car_state(label, front_center, back_center):
    if front_center and back_center:
        dx = front_center[0] - back_center[0]
        dy = front_center[1] - back_center[1]
        dist_px = math.sqrt(dx**2 + dy**2)
        dist_cl = px_to_car_lengths(dist_px)
        angle = math.atan2(dy, dx)

        if dist_cl is not None:
            print(f"[{label}] Pos: ({front_center[0]}, {front_center[1]})  "
                  f"Heading: {math.degrees(angle):.1f} deg  "
                  f"Dot sep: {dist_px:.1f}px = {dist_cl:.2f} car lengths")
        else:
            print(f"[{label}] Pos: ({front_center[0]}, {front_center[1]})  "
                  f"Heading: {math.degrees(angle):.1f} deg  "
                  f"Dot sep: {dist_px:.1f}px (not calibrated)")


def main(show_mask=False):
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    # Car 1: green (front) + blue (back)
    last_green, last_blue = None, None
    # Car 2: pink (front) + yellow (back)
    last_pink, last_yellow = None, None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        green_center,  green_mask  = find_color_centroid(hsv, GREEN_LOWER,  GREEN_UPPER)
        blue_center,   blue_mask   = find_color_centroid(hsv, BLUE_LOWER,   BLUE_UPPER)
        pink_center,   pink_mask   = find_color_centroid(hsv, PINK_LOWER,   PINK_UPPER)
        yellow_center, yellow_mask = find_color_centroid(hsv, YELLOW_LOWER, YELLOW_UPPER)

        if green_center:  last_green  = green_center
        if blue_center:   last_blue   = blue_center
        if pink_center:   last_pink   = pink_center
        if yellow_center: last_yellow = yellow_center

        # Car 1: green front, blue back
        frame = draw_car_overlay(
            frame,
            front_center=last_green,
            back_center=last_blue,
            label="GBCar1",           # front letter, back letter, then display name
            dot_color_front=(0, 255, 0),    # green
            dot_color_back=(255, 100, 0),   # blue (BGR)
            text_y_offset=0
        )

        # Car 2: pink front, yellow back
        frame = draw_car_overlay(
            frame,
            front_center=last_pink,
            back_center=last_yellow,
            label="PYCar2",
            dot_color_front=(180, 105, 255),  # pink (BGR)
            dot_color_back=(0, 215, 255),     # yellow (BGR)
            text_y_offset=35
        )

        print_car_state("Car1 G/B", last_green, last_blue)
        print_car_state("Car2 P/Y", last_pink,  last_yellow)

        if px_per_car is None:
            cv2.putText(frame, "Scale not calibrated - run bbox_calibrate.py",
                        (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 2)

        if show_mask:
            debug = cv2.hconcat([
                cv2.cvtColor(green_mask,  cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(blue_mask,   cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(pink_mask,   cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(yellow_mask, cv2.COLOR_GRAY2BGR),
            ])
            cv2.imshow("Masks: Green | Blue | Pink | Yellow", debug)

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