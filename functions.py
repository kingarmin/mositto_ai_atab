import cv2
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections
from PIL import Image
#model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
model_path='../model.pt'
print(model_path)
model = YOLO(model_path)
def face_detector():
    cap=cv2.VideoCapture(0)
    if not cap.isOpened():
        print("eror")
    else:
       while(1):
            ret, frame = cap.read()
            if ret:
                frame=cv2.flip(frame,1)
                output=model(frame)
                result = Detections.from_ultralytics(output[0])
                for i in result.xyxy:
                    x1 , y1 , x2 , y2 = map(int , i)
                    start_point=(x1 , y1)
                    end_point = (x2 , y2)
                    cv2.rectangle(frame , start_point , end_point , (0 , 0 , 255) , 2)
                cv2.imshow("AI",frame) 
            if cv2.getWindowProperty("AI", cv2.WND_PROP_VISIBLE) < 1:
                break  # Exit the loop when the window is closed


            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
       cap.release()
       cv2.destroyAllWindows()


def student_menue():
    face_detector()
def resizer(x,mode):
    from design import window_size
    if mode=='x':
        return (x*window_size['x'])/1000
    else:
        return (x*window_size['y'])/1000
def paths(x):
    from design import os 
    return os.path.join(os.getcwd(),x)

