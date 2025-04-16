import serial
import cv2
from ultralytics import YOLO


# arduinoData = serial.Serial('COM7', 9600)

#Convert the coordinates to a string and send them to Arduino
def send_coordinates_to_arduino(x, y):
    coordinates = f"{x},{y}\r"
    # arduinoData.write(coordinates.encode())
    print(f"Sent to Arduino: X{x} Y{y}")

#YOLO model using cvat
model_path = "Weapons-and-Knives-Detector-with-YOLOv8/runs/detect/Normal_Compressed/weights/best.pt"
yolo_model = YOLO(model_path)


capture = cv2.VideoCapture(0)  

while True:
    isTrue, frame = capture.read()
    if not isTrue:
        print("Error: Unable to read frame from the camera.")
        break

    # Perform knife detection using YOLOv8
    results = yolo_model(frame)

    knife_detected = False 

    for result in results:
        for box in result.boxes:
            xmin, ymin, xmax, ymax = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if conf >= 0.5 and yolo_model.names[cls].lower() == "knife":
                knife_detected = True

                # Draw rectangle around detected knife
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)  # Red color for knife detection
                label = f"{yolo_model.names[cls]} {conf:.2f}"
                cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                # Send coordinates of the knife's center to Arduino
                x_center = xmin + (xmax - xmin) // 2
                y_center = ymin + (ymax - ymin) // 2
                send_coordinates_to_arduino(x_center, y_center)

    if not knife_detected:
        print("No knife detected.")

    # Display the frame
    cv2.imshow('Knife Detection', frame)

    if cv2.waitKey(20) & 0xFF == ord('d'):
        break


capture.release()
cv2.destroyAllWindows()
