import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
from threading import Thread, Lock
import time
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
stream_url = 'rtsp://localhost:8554/stream'
fps = 25  # Frames per second (adjust based on stream)
buffer_duration = 10  # Buffer duration in seconds
frame_buffer = deque(maxlen=fps * 1000)
buffer_lock = Lock()  # Ensure thread-safe access to deque

# Initialize YOLO model
model = YOLO('yolov8n.pt')

# Vehicle detection parameters
car_class_id = 2
bus_class_id = 5
truck_class_id = 7
confidence_threshold = 0.5

# Area of Interest
Area_of_Interest = [(200, 210), (410, 210), (360, 330), (1, 320)]
avg_car_length = 4039.19  # mm
avg_car_width = 1572.89  # mm
previous_positions = {}
next_object_id = 0
interval_speeds = []  # Shared data structure for logging average speeds


def assign_id_to_new_object(cx, cy, previous_positions, threshold=50):
    global next_object_id
    for obj_id, (prev_x, prev_y) in previous_positions.items():
        if euclidean((cx, cy), (prev_x, prev_y)) < threshold:
            return obj_id
    obj_id = next_object_id
    next_object_id += 1
    return obj_id


def frame_grabber():
    """Thread to grab frames from the RTSP stream."""
    cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("Error: Could not open video stream")
        return

    print("Starting frame grabbing...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Add the frame to the deque
        with buffer_lock:
            frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            frame_buffer.append(frame)

        # Show the live stream (optional)
        cv2.imshow("RTSP Stream", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def frame_processor():
    """Thread to process frames for detection, speed estimation, and logging."""
    global previous_positions, next_object_id

    # Initialize data structures for logging
    car_logs = {}  # Tracks each car's cumulative speed and count
    interval_start_time = time.time()  # Start time for 10-second intervals
    timeout = 10
    last_frame_time = time.time()

    print("Starting frame processing...")
    while True:
        # Get the latest frame from the buffer
        with buffer_lock:
            if not frame_buffer:
                # Check for timeout if no frames are in the buffer
                if time.time() - last_frame_time > timeout:
                    print("Terminating due to no incoming data.")
                    break
                continue
            frame = frame_buffer[-1]  # Get the most recent frame from the deque

        last_frame_time = time.time()
        # Process the frame
        height, width = frame.shape[:2]
        print(f'Resized Frame Size: Width = {width}, Height = {height}')

        # Run YOLO detection
        results = model(frame)
        detections = results[0].boxes
        current_positions = {}
        cars_in_interval = 0  # Count cars detected in this interval

        for box in detections.data:
            x1, y1, x2, y2, conf, cls = box.tolist()
            cx = int((x1 + x2) // 2)
            cy = int((y1 + y2) // 2)

            if conf >= confidence_threshold and cls in [car_class_id, bus_class_id, truck_class_id]:
                # Check if the object is inside the area of interest
                result = cv2.pointPolygonTest(
                    np.array(Area_of_Interest, np.int32), (int(cx), int(cy)), False
                )

                if result >= 0:
                    # Assign unique ID
                    object_id = assign_id_to_new_object(cx, cy, previous_positions)
                    current_positions[object_id] = (cx, cy)
                    cars_in_interval += 1

                    # Draw the bounding box and ID
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID: {object_id}, Conf: {conf:.2f}", (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Calculate distance per pixel and speed
                    distance_per_pixel_x = (10) / (x2 - x1)  # meters/pixel
                    if object_id in previous_positions:
                        prev_x, prev_y = previous_positions[object_id]
                        pixel_distance = np.sqrt((cx - prev_x) ** 2 + (cy - prev_y) ** 2)
                        real_world_distance = pixel_distance * distance_per_pixel_x
                        speed = real_world_distance * fps  # Speed in m/s
                        speed_kmh = speed * 3.6  # Convert to km/h

                        # Log car speed
                        if object_id not in car_logs:
                            car_logs[object_id] = {"speed_sum": 0.0, "count": 0}
                        car_logs[object_id]["speed_sum"] += speed_kmh
                        car_logs[object_id]["count"] += 1

                        # Annotate speed on the frame
                        cv2.putText(frame, f'Speed: {speed * 3.6:.2f} km/h', (int(x1), int(y2) + 20), # Speed
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)
                        cv2.putText(frame, f'Distance Moved: {pixel_distance:.2f} px', (int(x1), int(y2) + 40), # Distance moved since last calculation in pixel
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)
                        cv2.putText(frame, f'Distance Moved: {real_world_distance:.2f} m', (int(x1), int(y2) + 60), # Distance moved since last calculation in meter
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)
                        cv2.putText(frame, f'Distance/Pixel: {distance_per_pixel_x:.2f} m', (int(x1), int(y2) + 80), # Distance per pixel in meter
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)

        # Update previous positions
        previous_positions = current_positions

        # 10-second interval processing
        if time.time() - interval_start_time >= buffer_duration:
            # Calculate the average speed for the interval
            total_speed = 0.0
            car_count = 0

            for car_id, data in car_logs.items():
                if data["count"] > 0:
                    average_speed = data["speed_sum"] / data["count"]
                    total_speed += average_speed
                    car_count += 1

            interval_average_speed = total_speed / car_count if car_count > 0 else 0.0
            with buffer_lock:  # Safely update shared data
                interval_speeds.append((interval_average_speed, car_count))
            print(f"Interval completed: Average Speed = {interval_average_speed:.2f} km/h, Cars = {car_count}")

            # Reset interval start time
            interval_start_time = time.time()

        # Draw the area of interest
        cv2.polylines(frame, [np.array(Area_of_Interest, np.int32)], True, (0, 255, 255), 2)

        # Display the processed frame
        cv2.imshow("Processed Stream", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Print final statistics after processing is done
    print("\nSummary of Interval Speeds:")
    for idx, (avg_speed, count) in enumerate(interval_speeds):
        print(f"Interval {idx + 1}: Average Speed = {avg_speed:.2f} km/h, Cars = {count}")


def visualize_speeds():
    """Thread to visualize real-time average speeds."""
    print("Starting visualization...")

    # Setup Matplotlib for live plotting
    fig, ax = plt.subplots()
    ax.set_title("Average Speed vs Time")
    ax.set_xlabel("Time Interval")
    ax.set_ylabel("Average Speed (km/h)")
    line, = ax.plot([], [], marker='o', label="Average Speed")
    ax.legend()

    def update_plot(frame):
        """Update the plot with the latest data."""
        with buffer_lock:  # Ensure thread-safe access to interval_speeds
            x_data = range(1, len(interval_speeds) + 1)
            y_data = [speed for speed, _ in interval_speeds]

        line.set_data(x_data, y_data)
        ax.set_xlim(0, len(x_data) + 1)
        ax.set_ylim(0, max(y_data) + 10 if y_data else 10)
        return line,

    # Use Matplotlib's FuncAnimation for real-time updates
    ani = FuncAnimation(fig, update_plot, interval=1000)
    plt.show()


if __name__ == "__main__":
    # Start threads
    grabber_thread = Thread(target=frame_grabber)
    processor_thread = Thread(target=frame_processor)
    visualization_thread = Thread(target=visualize_speeds)

    grabber_thread.start()
    processor_thread.start()
    visualization_thread.start()

    grabber_thread.join()
    processor_thread.join()
    visualization_thread.join()
