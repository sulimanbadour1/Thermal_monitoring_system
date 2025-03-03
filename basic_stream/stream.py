import cv2


def main():
    # Update camera_index if your camera is not at index 0
    camera_index = 1

    # Create a VideoCapture object
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Cannot open camera at index {camera_index}")
        return

    while True:
        # Read frame from the camera
        ret, frame = cap.read()

        # If frame reading was not successful, break the loop
        if not ret:
            print("Can't receive frame. Exiting...")
            break

        # Display the resulting frame
        cv2.imshow("FLIR E6390 (Webcam-like) Stream", frame)

        # If 'q' is pressed, exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the capture and close window
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
