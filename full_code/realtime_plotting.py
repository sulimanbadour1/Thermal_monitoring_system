import cv2
import numpy as np
import csv
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt


def bgr_to_rgb(color):
    """Convert a BGR tuple to normalized RGB tuple."""
    b, g, r = color
    return (r / 255.0, g / 255.0, b / 255.0)


def extract_color_temp_map(image, min_temp, max_temp, x1=306, y1=36, x2=315, y2=211):
    """
    Extract the color-to-temperature mapping from the scale bar region.
    Coordinates (x1, y1) to (x2, y2) should cover the vertical temperature scale.
    """
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
    # Ask the user for the scale parameters.
    try:
        min_temp = float(input("Enter the MIN temperature on the scale: "))
        max_temp = float(input("Enter the MAX temperature on the scale: "))
    except ValueError:
        print("Invalid input. Please enter numeric values for temperatures.")
        return

    # Ask the user for the sampling time (interval in seconds).
    try:
        sampling_interval = float(
            input("Enter the sampling time (in seconds) for capturing data: ")
        )
    except ValueError:
        print("Invalid input. Please enter a numeric value for sampling time.")
        return

    # Ask the user to enter the points file name or path.
    points_file = input("Enter the filename or path for the points file: ").strip()
    points = []  # will store tuples of (x, y, name)
    try:
        with open(points_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(",")
                    if len(parts) == 3:
                        try:
                            x = int(parts[0].strip())
                            y = int(parts[1].strip())
                            name = parts[2].strip()
                            points.append((x, y, name))
                        except ValueError:
                            print(f"Skipping invalid line: {line}")
                    elif len(parts) == 2:
                        try:
                            x = int(parts[0].strip())
                            y = int(parts[1].strip())
                            name = f"Point {len(points)+1}"
                            points.append((x, y, name))
                        except ValueError:
                            print(f"Skipping invalid line: {line}")
        if len(points) != 5:
            print(
                "Error: The file must contain exactly 5 points (each with x,y[,name])."
            )
            return
    except FileNotFoundError:
        print(f"Error: File '{points_file}' not found.")
        return

    # Create photos directory if it doesn't exist.
    photos_dir = "photos"
    if not os.path.exists(photos_dir):
        os.makedirs(photos_dir)

    # Open CSV file for logging temperature data.
    csv_filename = "photo_temperature_data.csv"
    csvfile = open(csv_filename, "w", newline="")
    writer = csv.writer(csvfile)
    header = ["Photo", "Timestamp"]
    for pt in points:
        header.append(f"Estimated Temperature {pt[2]} (C)")
    writer.writerow(header)

    # Setup camera. Change camera_index if needed.
    camera_index = 1  # Adjust as needed; often index 0 is the default camera.
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Cannot open camera at index {camera_index}")
        csvfile.close()
        return

    # Initialize variables for photo capture and plotting.
    last_capture_time = time.time()
    start_time = last_capture_time  # record when streaming started
    img_counter = 0

    # Initialize real-time plotting in interactive mode.
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Temperature (C)")
    ax.set_title("Real-Time Temperature Data")
    times_data = []
    # Create a list for each of the 5 points.
    temps_for_points = [[] for _ in range(5)]
    plot_lines = [ax.plot([], [], marker="o", label=points[i][2])[0] for i in range(5)]
    ax.legend()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame. Exiting...")
            break

        # Get frame height for overlay text.
        height = frame.shape[0]
        current_time = time.time()
        elapsed_time = current_time - start_time

        # Draw overlay text.
        overlay_text1 = f"Time Elapsed: {elapsed_time:.1f} s"
        overlay_text2 = f"Photos Captured: {img_counter}"
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

        # Pinpoint the points of interest by drawing circles and names.
        for x, y, name in points:
            cv2.circle(frame, (x, y), 5, (0, 0, 255), 2)  # red circle with radius 5
            cv2.putText(
                frame,
                name,
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
                cv2.LINE_AA,
            )

        # Display the live video stream.
        cv2.imshow("FLIR E6390 (Webcam-like) Stream", frame)

        # Capture photo and process every sampling_interval seconds.
        if current_time - last_capture_time >= sampling_interval:
            photo_filename = f"image_{img_counter:03d}.png"
            photo_path = os.path.join(photos_dir, photo_filename)
            cv2.imwrite(photo_path, frame)

            try:
                # Process the frame to estimate the temperature.
                color_temp_map = extract_color_temp_map(frame, min_temp, max_temp)
            except ValueError as e:
                print(f"Error processing image: {e}")
                color_temp_map = None

            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            temps = []
            if color_temp_map is not None:
                for idx, (x, y, name) in enumerate(points):
                    try:
                        estimated_temp = estimate_temperature(
                            frame, color_temp_map, x, y
                        )
                    except ValueError as e:
                        print(f"Error processing {name} in image {photo_filename}: {e}")
                        estimated_temp = np.nan
                    temps.append(estimated_temp)
                print(
                    f"Captured {photo_filename} at {timestamp_str} with temperatures: {', '.join([f'{t:.2f}' if not np.isnan(t) else 'Error' for t in temps])}"
                )
            else:
                temps = [np.nan] * len(points)
                print(
                    f"Captured {photo_filename} at {timestamp_str} but failed to estimate temperatures."
                )

            # Log the data into CSV.
            csv_row = [photo_filename, timestamp_str]
            for temp in temps:
                if np.isnan(temp):
                    csv_row.append("Error")
                else:
                    csv_row.append(f"{temp:.2f}")
            writer.writerow(csv_row)

            # Update real-time plot data.
            times_data.append(elapsed_time)
            for i in range(5):
                temps_for_points[i].append(temps[i])
                # Update the existing plot line data instead of redrawing the whole plot.
                plot_lines[i].set_data(times_data, temps_for_points[i])
            ax.relim()
            ax.autoscale_view()

            plt.draw()
            plt.pause(0.001)

            img_counter += 1
            last_capture_time = current_time

        # Allow user to quit the application by pressing 'q'.
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Clean up.
    cap.release()
    cv2.destroyAllWindows()
    csvfile.close()
    plt.ioff()
    plt.show()
    print(f"Temperature data saved to '{csv_filename}'.")


if __name__ == "__main__":
    main()
