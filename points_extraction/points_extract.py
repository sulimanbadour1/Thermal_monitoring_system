import cv2

# Global list to store the selected points
points = []


def mouse_callback(event, x, y, flags, param):
    global points
    # Capture left mouse button clicks if fewer than 5 points have been selected
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 5:
        points.append((x, y))
        print(f"Point selected: ({x}, {y})")


def main():
    global points
    camera_index = 1  # Update the index if your camera is different
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Cannot open camera at index {camera_index}")
        return

    window_name = "FLIR E6390 (Webcam-like) Stream"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame. Exiting...")
            break

        # Draw each selected point on the frame with a circle and coordinates
        for point in points:
            cv2.circle(frame, point, 5, (0, 255, 0), -1)  # Draw a filled green circle
            cv2.putText(
                frame,
                f"{point}",
                (point[0] + 10, point[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

        cv2.imshow(window_name, frame)

        # If five points are selected, print them, save to file, and break the loop
        if len(points) == 5:
            print("\nFive points selected:")
            for idx, point in enumerate(points, 1):
                print(f"Point {idx}: {point}")

            # Save points to a text file
            with open("points.txt", "w") as file:
                for point in points:
                    file.write(f"{point[0]}, {point[1]}\n")
            print("Points saved to points.txt")

            # Exit the loop after saving the points
            break

        # Allow exit by pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
