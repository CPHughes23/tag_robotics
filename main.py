import cv2
import cv2.aruco as aruco
import numpy as np
import time
import matplotlib.pyplot as plt
import random

def draw_two_axes(img, camera_matrix, dist_coeffs, rvec, tvec, length: float = 0.05):
    # 3D points: origin, X, Y
    axis_points = np.array([
        [0.0, 0.0, 0.0],
        [length, 0.0, 0.0],
        [0.0, length, 0.0],
    ], dtype=np.float32)

    imgpts, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs)

    origin = tuple(imgpts[0].ravel().astype(int))
    x_axis = tuple(imgpts[1].ravel().astype(int))
    y_axis = tuple(imgpts[2].ravel().astype(int))

    cv2.line(img, origin, x_axis, (0, 0, 255), 3)  # X axis in red
    cv2.line(img, origin, y_axis, (0, 255, 0), 3)  # Y axis in green

    return img

def main():
    plt.ion() # Turn on interactive mode
    fig, ax = plt.subplots()
    x, y = [], []
    hl, = ax.plot([], [], 'ro')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    # Camera calibration (example values)
    camera_matrix = np.array([[800, 0, 320],
                              [0, 800, 240],
                              [0, 0, 1]], dtype=np.float32)
    
    dist_coeffs = np.zeros((5, 1), dtype=np.float32)

    marker_length = 0.05  # meters

    # 3D object points of a square marker relative to its center
    obj_points = np.array([
        [-marker_length/2,  marker_length/2, 0],
        [ marker_length/2,  marker_length/2, 0],
        [ marker_length/2, -marker_length/2, 0],
        [-marker_length/2, -marker_length/2, 0]
    ], dtype=np.float32)

    # Open video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    # Setup ArUco detector
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    aruco_params = aruco.DetectorParameters()
    aruco_params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX  # smoother corners
    detector = aruco.ArucoDetector(aruco_dict, aruco_params)

    # Dictionaries to store last known poses per marker ID
    last_rvec = {}
    last_tvec = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = detector.detectMarkers(gray)

        if ids is not None:
            # Draw detected marker outlines
            aruco.drawDetectedMarkers(frame, corners, ids)

            for i in range(len(ids)):
                marker_id = int(ids[i][0])

                # SolvePnP for this marker
                success, rvec, tvec = cv2.solvePnP(
                    obj_points, corners[i], camera_matrix, dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE
                )

                if success:
                    # Update last known pose
                    last_rvec[marker_id] = rvec
                    last_tvec[marker_id] = tvec
                else:
                    # If detection fails, fallback to last known pose
                    if marker_id in last_rvec:
                        rvec = last_rvec[marker_id]
                        tvec = last_tvec[marker_id]
                    else:
                        continue  # no pose to draw yet

                # Draw axes
                draw_two_axes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)

        # Display frame
        cv2.imshow("ArUco Tracking", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

        # print("rvec: ", rvec)
        print("tvec: ", tvec[0], tvec[1])


        # 1. Update data

        # 2. Update the plot line
        hl.set_data([tvec[0][0]], [tvec[1][0]])
        
        # 4. Redraw the canvas
        fig.canvas.draw()
        fig.canvas.flush_events() # Process any GUI events

        # 5. Pause for a short duration to control update speed
        plt.pause(0.01) # Pause duration in seconds



    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()