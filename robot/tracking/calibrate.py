import cv2
import cv2.aruco as aruco
import numpy as np
import argparse
import glob
import os

def generate_charuco_board(out_file="charuco_board.png"):
    """Generates a Charuco board image you can print."""
    squares_x = 5
    squares_y = 7
    square_length = 0.04  # meters
    marker_length = 0.02  # meters

    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    board = aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, aruco_dict)

    img = board.generateImage((600, 800))
    cv2.imwrite(out_file, img)
    print(f"Charuco board image saved to {out_file}. Print it for calibration.")

def capture_calibration_images(save_dir="calib_images"):
    """Opens the camera, lets you capture frames by pressing SPACE, quit with Q."""
    os.makedirs(save_dir, exist_ok=True)

    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    board = aruco.CharucoBoard((5, 7), 0.04, 0.02, aruco_dict)
    detector = aruco.CharucoDetector(board)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open camera.")
        return

    count = 0
    print("Camera open. Press SPACE to capture a frame, Q to quit and proceed to calibration.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        charuco_corners, charuco_ids, marker_corners, _ = detector.detectBoard(gray)

        display = frame.copy()

        # Draw detected markers and corners so you can see what's being picked up
        if marker_corners:
            aruco.drawDetectedMarkers(display, marker_corners)
        if charuco_ids is not None and len(charuco_ids) >= 4:
            cv2.aruco.drawDetectedCornersCharuco(display, charuco_corners, charuco_ids)
            status = f"Detected {len(charuco_ids)} corners — good to capture"
            color = (0, 255, 0)
        else:
            status = "Not enough corners detected"
            color = (0, 0, 255)

        cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(display, f"Captured: {count}  |  SPACE=capture  Q=done", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Calibration Capture", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            if charuco_ids is not None and len(charuco_ids) >= 4:
                path = os.path.join(save_dir, f"calib_{count:03d}.jpg")
                cv2.imwrite(path, frame)
                print(f"Saved {path} ({len(charuco_ids)} corners)")
                count += 1
            else:
                print("Skipped — not enough corners detected in this frame.")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nCaptured {count} images to '{save_dir}/'.")
    return count

def calibrate_camera_from_charuco(images_glob="calib_images/*.jpg", save_file="camera_params.npz"):
    """Calibrates camera using images of a printed Charuco board."""
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    board = aruco.CharucoBoard((5, 7), 0.04, 0.02, aruco_dict)
    detector = aruco.CharucoDetector(board)

    all_obj_points = []
    all_img_points = []
    image_size = None

    image_files = glob.glob(images_glob)
    if len(image_files) == 0:
        print("No images found for calibration. Make sure the path is correct.")
        return

    for fname in image_files:
        img = cv2.imread(fname)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = gray.shape[::-1]

        charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)

        if charuco_ids is not None and len(charuco_ids) >= 4:
            obj_pts, img_pts = board.matchImagePoints(charuco_corners, charuco_ids)
            if obj_pts is not None and img_pts is not None:
                all_obj_points.append(obj_pts)
                all_img_points.append(img_pts)

    if len(all_obj_points) == 0:
        print("No Charuco corners detected in any images. Cannot calibrate.")
        return

    print(f"Calibrating using {len(all_obj_points)} valid images...")

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        all_obj_points,
        all_img_points,
        image_size,
        None,
        None,
        flags=cv2.CALIB_RATIONAL_MODEL
    )

    print(f"Calibration RMS reprojection error: {ret:.4f} px")
    print("Camera matrix:\n", camera_matrix)
    print("Distortion coefficients:\n", dist_coeffs.ravel())

    np.savez(save_file, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
    print(f"Camera parameters saved to {save_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true",
                        help="Generate a Charuco board image to print.")
    parser.add_argument("--capture", action="store_true",
                        help="Open camera to capture calibration images, then calibrate.")
    parser.add_argument("--images", type=str, default="calib_images/*.jpg",
                        help="Glob pattern for calibration images (used without --capture).")
    parser.add_argument("--output", type=str, default="camera_params.npz",
                        help="File to save camera parameters.")
    args = parser.parse_args()

    if args.generate:
        generate_charuco_board()
    elif args.capture:
        count = capture_calibration_images("calib_images")
        if count and count > 0:
            calibrate_camera_from_charuco("calib_images/*.jpg", args.output)
    else:
        calibrate_camera_from_charuco(args.images, args.output)

if __name__ == '__main__':
    main()