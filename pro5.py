import cv2
import os
import time
from ultralytics import YOLO

# Initialize YOLO model
working_directory = os.getcwd()
model = YOLO(os.path.join(working_directory, ".node-red/final.pt"))

# Function to process video
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    output = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        for result in results:
            json_result = result.tojson(normalize=False)
            detected = json_result[18:23]
            if len(detected) > 1:
                return [True,detected]  # Detected
    return [False]  # Not detected

# Main function
def main():
    video_capture = cv2.VideoCapture(0)

    working_directory = os.getcwd()
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video = cv2.VideoWriter(os.path.join(working_directory, ".node-red/output.avi"), fourcc, 20.0, (640, 480))

    start_time = time.time()
    while (time.time() - start_time) < 3:  # Record for 3 seconds
        ret, frame = video_capture.read()
        if ret:
            output_video.write(frame)
            # Display the captured frame continuously (within the loop)
            cv2.imshow("Camera Feed", frame)
            # Handle key press for early exit (optional)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Close windows after recording finishes
    cv2.destroyAllWindows()

    video_path = '.node-red/output.avi'
    object_detected = process_video(video_path)
    
    return object_detected

# Call the main function
object_detected = main()

# Send forward true/false
if object_detected[0]:
    print("Object Detected: True")
    print(f'{object_detected[1]} FOUND!!!!!')
else:
    print("Object Detected: False")
