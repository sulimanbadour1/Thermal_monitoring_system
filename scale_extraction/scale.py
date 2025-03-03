import cv2
import numpy as np
import csv


def main():
    """
    1) Load a thermal image with an overlaid scale bar.
    2) Crop the scale bar using fixed coordinates.
    3) Ask user for min and max temperature values.
    4) Map each row's average color (BGR) to an interpolated temperature.
    5) Save the row-by-row data to a CSV file.
    """

    # --- 1. LOAD THE IMAGE ---
    # Change this path to your actual file location
    image_path = r"E:\Novak_part_time_job\Thermal\photos\image_000.png"
    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Could not read image from {image_path}")
        return

    # --- 2. CROP THE SCALE BAR ---
    # Coordinates for the scale bar (top-left -> (306, 36), bottom-right -> (315, 211))
    x1, y1 = 306, 36
    x2, y2 = 315, 211
    cropped_scale = image[y1:y2, x1:x2]

    # Save cropped scale bar for verification (optional)
    cv2.imwrite("cropped_scale_bar.png", cropped_scale)
    print("Cropped scale bar saved as 'cropped_scale_bar.png'.")

    # --- 3. ASK USER FOR MIN/MAX TEMPERATURE ---
    min_temp = float(input("Enter the MIN temperature on the scale: "))
    max_temp = float(input("Enter the MAX temperature on the scale: "))

    # --- 4. BUILD COLOR-TO-TEMPERATURE MAPPING ---
    scale_height, scale_width, _ = cropped_scale.shape
    color_temp_map = []  # Will hold tuples of (row_index, temperature, [B, G, R])

    for row_index in range(scale_height):
        # Extract the row's pixels
        row_pixels = cropped_scale[row_index, :, :]  # shape: [scale_width, 3]

        # Compute average BGR for this row
        avg_bgr = np.mean(row_pixels, axis=0)  # => [B, G, R]
        avg_bgr = avg_bgr.tolist()

        # Interpolate temperature (top row = max_temp, bottom row = min_temp)
        if scale_height > 1:
            fraction = row_index / (scale_height - 1)
        else:
            fraction = 0  # Edge case if there's only 1 row
        row_temp = max_temp - fraction * (max_temp - min_temp)

        # Store data: (row_index, temperature, BGR)
        color_temp_map.append((row_index, row_temp, avg_bgr))

    # --- 5. SAVE TO CSV ---
    csv_filename = "color_temp_map.csv"
    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write a header row
        writer.writerow(["RowIndex", "Temperature_C", "B", "G", "R"])
        # Write each row of data
        for row_idx, temp, bgr in color_temp_map:
            b, g, r = bgr
            writer.writerow(
                [row_idx, f"{temp:.2f}", f"{b:.2f}", f"{g:.2f}", f"{r:.2f}"]
            )

    print(f"Saved {len(color_temp_map)} rows of data to '{csv_filename}'.")

    # (Optional) Print first few entries
    print("\nFirst 5 rows in the CSV:")
    for i, (row_idx, temp, bgr) in enumerate(color_temp_map[:5]):
        b, g, r = bgr
        print(f"Row {row_idx}, Temp={temp:.2f}°C, B={b:.2f}, G={g:.2f}, R={r:.2f}")


if __name__ == "__main__":
    main()
