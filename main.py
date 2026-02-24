import cv2
import cv2.aruco as aruco
import numpy as np

def draw_two_axes(img, camera_matrix, dist_coeffs, rvec, tvec, length: float = 0.05):
    # 3D points: origin, X, Y
    axis_points = np.float32([
        [0, 0, 0],          # origin
        [length, 0, 0],     # X axis
        [0, length, 0],     # Y axis
    ])

    imgpts, _ = cv2.projectPoints(
        axis_points, rvec, tvec, camera_matrix, dist_coeffs
    )

    origin = tuple(imgpts[0].ravel().astype(int))
    x_axis = tuple(imgpts[1].ravel().astype(int))
    y_axis = tuple(imgpts[2].ravel().astype(int))

    cv2.line(img, origin, x_axis, (0,0,255), 3)  # X (red)
    cv2.line(img, origin, y_axis, (0,255,0), 3)  # Y (green)

    return img


def main():
    camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((5, 1)) # Assuming no distortion for this example

    marker_length = 0.05  # Side length in meters (e.g., 5cm)

    obj_points = np.array([
        [-marker_length/2,  marker_length/2, 0],
        [ marker_length/2,  marker_length/2, 0],
        [ marker_length/2, -marker_length/2, 0],
        [-marker_length/2, -marker_length/2, 0]
    ], dtype=np.float32)

    # Create a VideoCapture object (0 for built-in webcam, 1 or more for external USB cameras)
    cap = cv2.VideoCapture(0)

    # Setup ArUco dictionary and detector parameters
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    aruco_params = aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        # Capture frame-by-frame; ret is a boolean, frame is the image array
        ret, frame = cap.read()

        # If frame is read correctly, ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Detect markers
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = detector.detectMarkers(gray)

        # Draw detected markers
        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)

            for i in range(len(ids)):
                # Solve PnP for this marker
                _, rvec, tvec = cv2.solvePnP(
                    obj_points, corners[i], camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE
                )

                # Draw axes for this marker
                draw_two_axes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)

        # Display the resulting frame
        cv2.imshow("ArUco Tracking", frame)
        if cv2.waitKey(1) & 0xFF == 27: break # ESC to quit

    # Release the capture and destroy all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()