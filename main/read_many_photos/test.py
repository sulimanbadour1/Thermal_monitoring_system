import cv2
import numpy as np
import csv
import os
import time
from datetime import datetime


def extract_color_temp_map(image, min_temp, max_temp, x1=306, y1=36, x2=315, y2=211):
    """
    Extracts a color-to-temperature mapping from the cropped scale bar region.
    The scale bar is assumed to run vertically from (x1, y1) to (x2, y2).
    """
    cropped_scale = image[y1:y2, x1:x2]
    scale_height, scale_width, _ = cropped_scale.shape
    color_temp_map = []

    # For each row in the scale bar, compute the average BGR color and
    # map the row to a temperature by linear interpolation.
    for row_index in range(scale_height):
        row_pixels = cropped_scale[row_index, :, :]
        avg_bgr = np.mean(row_pixels, axis=0).tolist()  # [B, G, R]
        fraction = row_index / (scale_height - 1) if scale_height > 1 else 0
        row_temp = max_temp - fraction * (max_temp - min_temp)
        color_temp_map.append((row_temp, avg_bgr))

    # Sort by ascending temperature (optional, as our interpolation already
    # goes from top (max_temp) to bottom (min_temp))
    color_temp_map.sort(key=lambda x: x[0])
    return color_temp_map


def estimate_temperature(image, color_temp_map, x_target, y_target):
    """
    Estimates the temperature at a target (x,y) coordinate in the image.
    The target pixel’s BGR color is compared to each row of the scale using
    Euclidean distance in RGB space.
    """
    # Get the BGR color at the target point and convert to normalized RGB
    target_color = image[y_target, x_target, :]  # note: image[y, x]
    b, g, r = target_color
    target_rgb = (r / 255.0, g / 255.0, b / 255.0)

    min_distance = float("inf")
    estimated_temp = None
    # Compare with each row in the color_temp_map
    for temp, bgr in color_temp_map:
        tb, tg, tr = bgr
        row_rgb = (tr / 255.0, tg / 255.0, tb / 255.0)
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
    photos_dir = r"E:\Novak_part_time_job\Thermal\point_temp\photos"
    if not os.path.exists(photos_dir):
        print("Photos directory not found!")
        return

    # Ask the user for scale min/max temperatures and the target point coordinates.
    try:
        min_temp = float(input("Enter the MIN temperature on the scale: "))
        max_temp = float(input("Enter the MAX temperature on the scale: "))
        x_target = int(input("Enter the x-coordinate of the target point: "))
        y_target = int(input("Enter the y-coordinate of the target point: "))
    except ValueError:
        print("Invalid input. Please enter numeric values.")
        return

    # Prepare the output CSV file.
    csv_filename = "photo_temperature_data.csv"
    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Photo", "Timestamp", "Estimated Temperature (°)"])

        # Loop through all image files in the photos directory.
        for photo_filename in sorted(os.listdir(photos_dir)):
            if photo_filename.lower().endswith((".png", ".jpg", ".jpeg")):
                photo_path = os.path.join(photos_dir, photo_filename)
                image = cv2.imread(photo_path)
                if image is None:
                    print(f"WARNING: Could not read image {photo_filename}. Skipping.")
                    continue

                # Build the color-to-temperature map from the scale bar.
                color_temp_map = extract_color_temp_map(image, min_temp, max_temp)
                # Estimate the temperature at the specified (x, y) point.
                estimated_temp = estimate_temperature(
                    image, color_temp_map, x_target, y_target
                )

                # Get the file's modification time as a proxy for capture timestamp.
                mod_time = os.path.getmtime(photo_path)
                timestamp_str = datetime.fromtimestamp(mod_time).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

                print(
                    f"Photo: {photo_filename}, Timestamp: {timestamp_str}, Estimated Temperature: {estimated_temp:.2f}"
                )
                writer.writerow(
                    [photo_filename, timestamp_str, f"{estimated_temp:.2f}"]
                )

    print(f"Temperature data has been saved to '{csv_filename}'.")


if __name__ == "__main__":
    main()
