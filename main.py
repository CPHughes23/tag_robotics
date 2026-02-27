import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# --- Tune these HSV ranges for your specific lighting conditions ---
# Use the HSV tuner script below if colors aren't detecting well
BLUE_LOWER = np.array([80, 51, 159])
BLUE_UPPER = np.array([115, 255, 255])

GREEN_LOWER = np.array([40, 80, 70])
GREEN_UPPER = np.array([80, 255, 255])

MIN_BLOB_AREA = 200  # Minimum pixel area to count as a valid detection (filters noise)


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


def draw_tracking_overlay(frame, blue_center, green_center):
    """
    Draws the detected dot positions, the midpoint, and an orientation arrow.
    """
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

    return frame


def main():
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_title("Car Position (pixels)")
    hl, = ax.plot([], [], 'ro', markersize=8)
    ax.set_xlim(0, 640)
    ax.set_ylim(0, 480)
    ax.invert_yaxis()  # Match image coordinate system

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    # Store last known positions so the plot doesn't blank when detection drops
    last_blue = None
    last_green = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        blue_center, blue_mask = find_color_centroid(hsv, BLUE_LOWER, BLUE_UPPER)
        green_center, green_mask = find_color_centroid(hsv, GREEN_LOWER, GREEN_UPPER)

        # Fall back to last known position if detection drops for a frame
        if blue_center:
            last_blue = blue_center
        if green_center:
            last_green = green_center

        frame = draw_tracking_overlay(frame, blue_center, green_center)

        # Print and plot midpoint when both dots are visible
        if last_blue and last_green:
            mid_x = (last_blue[0] + last_green[0]) / 2
            mid_y = (last_blue[1] + last_green[1]) / 2
            angle = math.atan2(last_green[1] - last_blue[1],
                               last_green[0] - last_blue[0])
            print(f"Position: ({mid_x:.1f}, {mid_y:.1f})  Heading: {math.degrees(angle):.1f} deg")

            hl.set_data([mid_x], [mid_y])
            fig.canvas.draw()
            fig.canvas.flush_events()

        # Show debug masks side by side (helpful for tuning HSV ranges)
        debug_masks = cv2.hconcat([
            cv2.cvtColor(blue_mask, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(green_mask, cv2.COLOR_GRAY2BGR)
        ])
        cv2.imshow("Masks: Blue | Green", debug_masks)
        cv2.imshow("Color Tracking", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

        plt.pause(0.01)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()