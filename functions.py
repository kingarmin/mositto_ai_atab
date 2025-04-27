import cv2
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections
from PIL import Image
from deepface import DeepFace
import numpy as np
import concurrent.futures
from time import time
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
#model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
model_path='../model.pt'
print(model_path)
model = YOLO(model_path)
faces=[]
def check_faces(x):
    global faces
    output=faces
    for i in range(len(faces)):
        result=DeepFace.verify(img1_path=faces[i] , img2_path=x)
        if result['verified']:
            return 0
    faces.append(x)
    return 1
def face_detector():
    cap=cv2.VideoCapture(0)
    if not cap.isOpened():
        print("eror")
    else:
       start=time();
       while(time()-start<=15):
            ret, frame = cap.read()
            if ret:
                frame=cv2.flip(frame,1)
                output=model(frame)
                result = Detections.from_ultralytics(output[0])
                if(len(list(map(int,result.area)))!=0):
                    for i in result.xyxy:
                        x1 , y1 , x2 , y2 = map(int , i)
                        start_point=(x1 , y1)
                        end_point = (x2 , y2)
                        cv2.rectangle(frame , start_point , end_point , (255 , 0 , 0) , 2)
                        #check_faces(frame[y1:y2 , x1:x2])
                        face_crop = frame[y1:y2, x1:x2].copy()
                        executor.submit(check_faces, face_crop)
                cv2.imshow("AI",frame) 
            if cv2.getWindowProperty("AI", cv2.WND_PROP_VISIBLE) < 1:
                break  # Exit the loop when the window is closed


            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
       cap.release()
       cv2.destroyAllWindows()
    for idx, crop in enumerate(faces):
        if crop.size != 0:  # Make sure crop is valid
            cv2.imshow(f"Crop {idx}", crop)
            key = cv2.waitKey(0)  # Wait until any key is pressed
            if key == ord('q'):   # If 'q' pressed, exit early
                break

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

