import cv2
import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def main():
    # -------------------------------------------------------------------------
    # 1) LOAD AND CROP THE THERMAL IMAGE (same approach as before)
    # -------------------------------------------------------------------------
    image_path = r"E:\Novak_part_time_job\Thermal\photos\image_000.png"
    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Could not read image from {image_path}")
        return

    # Coordinates for the scale bar (example)
    x1, y1 = 306, 36
    x2, y2 = 315, 211
    cropped_scale = image[y1:y2, x1:x2]

    # -------------------------------------------------------------------------
    # 2) ASK USER FOR MIN/MAX TEMPERATURE AND BUILD color_temp_map
    # -------------------------------------------------------------------------
    min_temp = float(input("Enter the MIN temperature on the scale: "))
    max_temp = float(input("Enter the MAX temperature on the scale: "))

    scale_height, scale_width, _ = cropped_scale.shape
    color_temp_map = []  # will hold (temperature, [B, G, R])

    for row_index in range(scale_height):
        row_pixels = cropped_scale[row_index, :, :]
        avg_bgr = np.mean(row_pixels, axis=0).tolist()  # [B, G, R]

        # Interpolate temperature from top = max_temp to bottom = min_temp
        fraction = row_index / (scale_height - 1) if scale_height > 1 else 0
        row_temp = max_temp - fraction * (max_temp - min_temp)

        color_temp_map.append((row_temp, avg_bgr))

    # -------------------------------------------------------------------------
    # 3) CREATE A CUSTOM MATPLOTLIB COLORMAP FROM color_temp_map
    # -------------------------------------------------------------------------
    # color_temp_map is a list of (temp, [B, G, R]) from top (max_temp) to bottom (min_temp).
    # We want to:
    #   1) Sort by temperature (descending or ascending)
    #   2) Normalize temperature to [0, 1]
    #   3) Convert BGR to normalized RGB
    #   4) Create a colormap

    # 3a. Sort by ascending temperature if needed
    color_temp_map.sort(key=lambda x: x[0])  # sorts by temperature ascending

    # 3b. Extract separate lists for temperature and BGR
    temps = [item[0] for item in color_temp_map]  # e.g. [5.0, 6.0, ..., 42.7]
    bgrs = [item[1] for item in color_temp_map]  # e.g. [[10, 50, 200], ...]

    # 3c. Normalize temperature range to [0, 1]
    tmin, tmax = min(temps), max(temps)
    temp_norm = [(t - tmin) / (tmax - tmin) if (tmax - tmin) != 0 else 0 for t in temps]

    # 3d. Convert BGR to normalized RGB
    # OpenCV images are typically BGR in [0..255]. Matplotlib expects RGB in [0..1].
    rgbs = []
    for bgr in bgrs:
        b, g, r = bgr
        # Convert to 0..1 range
        r_norm = r / 255.0
        g_norm = g / 255.0
        b_norm = b / 255.0
        rgbs.append((r_norm, g_norm, b_norm))

    # 3e. Build a list of (normalized_temp, (r, g, b)) for the colormap
    cdict = list(zip(temp_norm, rgbs))
    # e.g. [ (0.0, (1.0, 0.0, 0.0)), (0.1, (1.0, 0.2, 0.1)), ... , (1.0, (0.0, 0.0, 1.0)) ]

    # 3f. Create the colormap
    # The 'from_list' constructor takes a list of (position, color). 'position' must be in [0..1].
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("thermal_cmap", cdict)

    # -------------------------------------------------------------------------
    # 4) VISUALIZE THE CUSTOM COLORMAP
    # -------------------------------------------------------------------------
    # We'll just create a simple gradient image to display the colormap.
    gradient = np.linspace(0, 1, 256)  # 256 steps
    gradient = np.vstack(
        (gradient, gradient)
    )  # shape: (2, 256) so we can imshow it as a strip

    plt.figure(figsize=(6, 2))
    plt.imshow(gradient, aspect="auto", cmap=custom_cmap)
    plt.title("Custom Thermal Colormap")
    plt.colorbar(label="Normalized Temperature")
    plt.show()

    # -------------------------------------------------------------------------
    # 5) (OPTIONAL) SAVE TO CSV
    # -------------------------------------------------------------------------
    # If you also want to save the row data to CSV (row_index, temp, B, G, R):
    import csv

    csv_filename = "color_temp_map.csv"
    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["RowIndex", "Temperature", "B", "G", "R"])
        for row_index, (temp, bgr) in enumerate(color_temp_map):
            b, g, r = bgr
            writer.writerow(
                [row_index, f"{temp:.2f}", f"{b:.2f}", f"{g:.2f}", f"{r:.2f}"]
            )
    print(f"Saved row-based color map to '{csv_filename}'.")


if __name__ == "__main__":
    main()
