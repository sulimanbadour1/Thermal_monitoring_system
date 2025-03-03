import cv2

# 1. Use the file path in a variable
image_path = r"E:\Novak_part_time_job\Thermal\photos\image_001.png"

# 2. Actually read the image into a NumPy array
image = cv2.imread(image_path)

# 3. Now image is an array, so you can slice it:
x1, y1 = 306, 36
x2, y2 = 315, 211

# 4. Crop
cropped = image[y1:y2, x1:x2]

cv2.imwrite("cropped_scale_bar.png", cropped)
