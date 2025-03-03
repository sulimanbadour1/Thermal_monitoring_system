import cv2
import os
import time


def main():
    # Create the photos directory if it doesn't exist
    photos_dir = "photos"
    if not os.path.exists(photos_dir):
        os.makedirs(photos_dir)

    # Update camera_index if your camera is not at index 1
    camera_index = 1

    # Create a VideoCapture object
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Cannot open camera at index {camera_index}")
        return

    photos_to_capture = 5  # Change this number to capture more or fewer photos
    img_counter = 0

    # Record the time of the last capture
    last_capture_time = time.time()

    while True:
        # Read frame from the camera
        ret, frame = cap.read()

        # If frame reading was not successful, break the loop
        if not ret:
            print("Can't receive frame. Exiting...")
            break

        # Display the resulting frame
        cv2.imshow("FLIR E6390 (Webcam-like) Stream", frame)

        # Check if 2 seconds have passed since the last capture
        current_time = time.time()
        if current_time - last_capture_time >= 2:
            img_name = os.path.join(photos_dir, f"image_{img_counter:03d}.png")
            cv2.imwrite(img_name, frame)
            print(f"Captured image: {img_name}")
            img_counter += 1
            last_capture_time = current_time

            # If we've captured the desired number of photos, exit automatically
            if img_counter >= photos_to_capture:
                break

        # Optionally, allow the user to exit early by pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the capture and close window
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
