import cv2
import numpy as np

def nothing(x):
    pass

def tune_hue(specified_color=""):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    cv2.namedWindow("HSV Tuner")

    # Create trackbars for lower and upper HSV bounds
    cv2.createTrackbar("H Lower", "HSV Tuner", 0,   179, nothing)
    cv2.createTrackbar("H Upper", "HSV Tuner", 179, 179, nothing)
    cv2.createTrackbar("S Lower", "HSV Tuner", 0,   255, nothing)
    cv2.createTrackbar("S Upper", "HSV Tuner", 255, 255, nothing)
    cv2.createTrackbar("V Lower", "HSV Tuner", 0,   255, nothing)
    cv2.createTrackbar("V Upper", "HSV Tuner", 255, 255, nothing)

    print("Adjust the sliders until only your target color shows as white in the mask.")
    print("Press 'p' to print the current values to copy into your tracking script.")
    print("Press ESC to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Read trackbar positions
        h_low  = cv2.getTrackbarPos("H Lower", "HSV Tuner")
        h_high = cv2.getTrackbarPos("H Upper", "HSV Tuner")
        s_low  = cv2.getTrackbarPos("S Lower", "HSV Tuner")
        s_high = cv2.getTrackbarPos("S Upper", "HSV Tuner")
        v_low  = cv2.getTrackbarPos("V Lower", "HSV Tuner")
        v_high = cv2.getTrackbarPos("V Upper", "HSV Tuner")

        lower = np.array([h_low, s_low, v_low])
        upper = np.array([h_high, s_high, v_high])

        mask = cv2.inRange(hsv, lower, upper)
        result = cv2.bitwise_and(frame, frame, mask=mask)

        # Show current values on the frame
        label = f"H:[{h_low}-{h_high}] S:[{s_low}-{s_high}] V:[{v_low}-{v_high}]"
        cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        # Stack original, mask, and masked result for easy comparison
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        display = np.hstack([frame, mask_bgr, result])

        # Scale down if your screen is small
        display = cv2.resize(display, (0, 0), fx=0.75, fy=0.75)

        cv2.imshow("HSV Tuner", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('p'):
            return h_low, s_low, v_low, h_high, s_high, v_high

    cap.release()
    cv2.destroyAllWindows()

def main():
    hue_values = tune_hue()
    if hue_values:
        h_low, s_low, v_low, h_high, s_high, v_high = hue_values
        print("\n--- Copy these into color_tracking.py ---")
        print(f"LOWER = np.array([{h_low}, {s_low}, {v_low}])")
        print(f"UPPER = np.array([{h_high}, {s_high}, {v_high}])")
        print("-----------------------------------------\n")

if __name__ == '__main__':
    main()