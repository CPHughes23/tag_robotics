import cv2
import numpy as np
import json
import os

CONFIG_FILE = "./robot/camera_config.json"

def nothing(x):
    pass

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    return {
        "blue":  {"lower": [0, 0, 0],     "upper": [179, 255, 255]},
        "green": {"lower": [0, 0, 0],     "upper": [179, 255, 255]},
        "car_scale": {"width_px": None, "height_px": None, "pixels_per_car_length": None}
    }

def save_config(config):
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved to {CONFIG_FILE}")

def tune_hue(color_key="blue", specified_color=""):
    """
    Opens HSV tuner. Pre-loads existing values from config if available.
    Press 's' to save to config. Press ESC to quit without saving.
    """
    config = load_config()
    existing = config.get(color_key, {"lower": [0,0,0], "upper": [179,255,255]})

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return None

    window = f"HSV Tuner - {specified_color or color_key}"
    cv2.namedWindow(window)

    # Pre-load existing values
    cv2.createTrackbar("H Lower", window, existing["lower"][0], 179, nothing)
    cv2.createTrackbar("H Upper", window, existing["upper"][0], 179, nothing)
    cv2.createTrackbar("S Lower", window, existing["lower"][1], 255, nothing)
    cv2.createTrackbar("S Upper", window, existing["upper"][1], 255, nothing)
    cv2.createTrackbar("V Lower", window, existing["lower"][2], 255, nothing)
    cv2.createTrackbar("V Upper", window, existing["upper"][2], 255, nothing)

    print(f"Tuning '{color_key}'. Press 's' to save, ESC to quit.")

    result_values = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        h_low  = cv2.getTrackbarPos("H Lower", window)
        h_high = cv2.getTrackbarPos("H Upper", window)
        s_low  = cv2.getTrackbarPos("S Lower", window)
        s_high = cv2.getTrackbarPos("S Upper", window)
        v_low  = cv2.getTrackbarPos("V Lower", window)
        v_high = cv2.getTrackbarPos("V Upper", window)

        lower = np.array([h_low, s_low, v_low])
        upper = np.array([h_high, s_high, v_high])

        mask = cv2.inRange(hsv, lower, upper)
        result = cv2.bitwise_and(frame, frame, mask=mask)

        # Show current values on the frame
        label = f"H:[{h_low}-{h_high}] S:[{s_low}-{s_high}] V:[{v_low}-{v_high}]"
        cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(frame, "S=Save  ESC=Quit", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        # Stack original, mask, and masked result for easy comparison
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        display = np.hstack([frame, mask_bgr, result])

        # Scale down if your screen is small
        display = cv2.resize(display, (0, 0), fx=0.75, fy=0.75)
        cv2.imshow(window, display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC — quit without saving
            break
        elif key == ord('s'):  # Save to config
            result_values = (h_low, s_low, v_low, h_high, s_high, v_high)
            config[color_key] = {
                "lower": [h_low, s_low, v_low],
                "upper": [h_high, s_high, v_high]
            }
            save_config(config)
            print(f"  lower = [{h_low}, {s_low}, {v_low}]")
            print(f"  upper = [{h_high}, {s_high}, {v_high}]")
            break

    cap.release()
    cv2.destroyAllWindows()
    return result_values


def main():
    print("Which color to tune? (blue/green)")
    color_key = input("> ").strip().lower()
    if color_key not in ("blue", "green"):
        color_key = "blue"
    tune_hue(color_key=color_key, specified_color=color_key.capitalize())


if __name__ == '__main__':
    main()