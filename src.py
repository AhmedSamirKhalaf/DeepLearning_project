import cv2
from ultralytics import YOLO

choice = int (input("enter 1 for Yolo_v8 and 2 for Yolo_v11 : "))
while True :
    if(choice == 1):
        model = YOLO('Yolo_v8_weights.pt') 
    elif(choice == 2):
        model = YOLO('Yolo_v11_weights.pt')
    cap = cv2.VideoCapture(0)  
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame.")
                break

            
            results = model.predict(source=frame, conf=0.9, show=True)

    except KeyboardInterrupt:
        print("\nDetection stopped.")
        break

cap.release()
cv2.destroyAllWindows()
