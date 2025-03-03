import cv2
import numpy as np
import csv
import os
import time
from datetime import datetime


def bgr_to_rgb(color):
    """Convert a BGR tuple to normalized RGB tuple."""
    b, g, r = color
    return (r / 255.0, g / 255.0, b / 255.0)


def extract_color_temp_map(image, min_temp, max_temp, x1=306, y1=36, x2=315, y2=211):
    """
    Extract the color-to-temperature mapping from the scale bar region.
    Coordinates (x1, y1) to (x2, y2) should cover the vertical temperature scale.
    """
    # Ensure the crop coordinates are within the image bounds.
    height, width, _ = image.shape
    if x1 < 0 or x2 > width or y1 < 0 or y2 > height:
        raise ValueError("Scale bar coordinates are out of image bounds.")

    cropped_scale = image[y1:y2, x1:x2]
    scale_height, scale_width, _ = cropped_scale.shape
    color_temp_map = []

    for row_index in range(scale_height):
        row_pixels = cropped_scale[row_index, :, :]
        # Average over the width for this row.
        avg_bgr = np.mean(row_pixels, axis=0).tolist()
        fraction = row_index / (scale_height - 1) if scale_height > 1 else 0
        # Interpolate temperature from top (max_temp) to bottom (min_temp)
        row_temp = max_temp - fraction * (max_temp - min_temp)
        color_temp_map.append((row_temp, avg_bgr))

    # Sort the mapping by temperature (ascending order)
    color_temp_map.sort(key=lambda x: x[0])
    return color_temp_map


def estimate_temperature(image, color_temp_map, x_target, y_target):
    """
    Estimate the temperature at the specified (x, y) point.
    Compares the target pixel's normalized RGB to each row's color in the scale.
    """
    height, width, _ = image.shape
    if not (0 <= x_target < width and 0 <= y_target < height):
        raise ValueError("Target coordinates are out of image bounds.")

    target_color = image[y_target, x_target, :]
    target_rgb = bgr_to_rgb(target_color)

    min_distance = float("inf")
    estimated_temp = None
    for temp, bgr in color_temp_map:
        row_rgb = bgr_to_rgb(bgr)
        # Euclidean distance in RGB space.
        distance = np.sqrt(
            (target_rgb[0] - row_rgb[0]) ** 2
            + (target_rgb[1] - row_rgb[1]) ** 2
            + (target_rgb[2] - row_rgb[2]) ** 2
        )
        if distance < min_distance:
            min_distance = distance
            estimated_temp = temp
    return estimated_temp


def main():
    # Ask the user for the scale parameters and target coordinates.
    try:
        min_temp = float(input("Enter the MIN temperature on the scale: "))
        max_temp = float(input("Enter the MAX temperature on the scale: "))
        x_target = int(input("Enter the x-coordinate of the target point: "))
        y_target = int(input("Enter the y-coordinate of the target point: "))
    except ValueError:
        print("Invalid input. Please enter numeric values.")
        return

    # Create photos directory if it doesn't exist.
    photos_dir = "photos"
    if not os.path.exists(photos_dir):
        os.makedirs(photos_dir)

    # Open CSV file for logging temperature data.
    csv_filename = "photo_temperature_data.csv"
    csvfile = open(csv_filename, "w", newline="")
    writer = csv.writer(csvfile)
    writer.writerow(["Photo", "Timestamp", "Estimated Temperature (°)"])

    # Setup camera. Change camera_index if needed.
    camera_index = 1  # Adjust as needed; often index 0 is the default camera.
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Cannot open camera at index {camera_index}")
        csvfile.close()
        return

    photo_capture_interval = 2  # seconds between captures; adjust as needed.
    last_capture_time = time.time()
    start_time = last_capture_time  # record when streaming started
    img_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame. Exiting...")
            break

        # Get frame height to place text.
        height = frame.shape[0]
        current_time = time.time()
        elapsed_time = current_time - start_time
        overlay_text1 = f"Time Elapsed: {elapsed_time:.1f} s"
        overlay_text2 = f"Photos Captured: {img_counter}"

        # Draw overlay text.
        cv2.putText(
            frame,
            overlay_text1,
            (10, height - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            overlay_text2,
            (10, height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

        # Display the live video stream.
        cv2.imshow("FLIR E6390 (Webcam-like) Stream", frame)

        # Capture photo every photo_capture_interval seconds.
        if current_time - last_capture_time >= photo_capture_interval:
            photo_filename = f"image_{img_counter:03d}.png"
            photo_path = os.path.join(photos_dir, photo_filename)
            cv2.imwrite(photo_path, frame)

            try:
                # Process the frame to estimate the temperature.
                color_temp_map = extract_color_temp_map(frame, min_temp, max_temp)
                estimated_temp = estimate_temperature(
                    frame, color_temp_map, x_target, y_target
                )
            except ValueError as e:
                print(f"Error processing image: {e}")
                estimated_temp = None

            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if estimated_temp is not None:
                print(
                    f"Captured {photo_filename} at {timestamp_str} with estimated temperature {estimated_temp:.2f}"
                )
                writer.writerow(
                    [photo_filename, timestamp_str, f"{estimated_temp:.2f}"]
                )
            else:
                print(
                    f"Captured {photo_filename} at {timestamp_str} but failed to estimate temperature."
                )

            img_counter += 1
            last_capture_time = current_time

        # Allow user to quit the application by pressing 'q'.
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Clean up.
    cap.release()
    cv2.destroyAllWindows()
    csvfile.close()
    print(f"Temperature data saved to '{csv_filename}'.")


if __name__ == "__main__":
    main()
