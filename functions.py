import cv2
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections
from PIL import Image
from deepface import DeepFace
import numpy as np
import concurrent.futures
from time import time
import os
#model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
model_path='../model.pt'
print(model_path)
model = YOLO(model_path)
faces=[]
path_of_dataset=os.getcwd()+'/data/images'
data=os.listdir(path_of_dataset)
atab_stu={'present': []}
def check_faces(x):
    global faces
    for i in range(len(faces)):
        result=DeepFace.verify(img1_path=faces[i] , img2_path=x)
        if result['verified']:
            return 0
    faces.append(x)
    return 1
def face_detector():
    global faces
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    faces = []

    cap=cv2.VideoCapture(0)
    if not cap.isOpened():
        print("eror")
    else:
       start=time();
       while(time()-start<=5):
            ret, frame = cap.read()
            if ret:
                frame=cv2.flip(frame,1)
                output=model(frame)
                result = Detections.from_ultralytics(output[0])
                alpha_frame=frame.copy()
                if len(result.xyxy) > 0:  # Assuming result.xyxy contains detected bounding boxes
                    face_crop = []
                    for i in result.xyxy:
                        x1, y1, x2, y2 = map(int, i)  # Extract bounding box coordinates
                        start_point = (x1, y1)
                        end_point = (x2, y2)

                        # Draw rectangle around the face
                        cv2.rectangle(alpha_frame, start_point, end_point, (255, 0, 0), 2)

                        # Crop the face region (without excessive padding)
                        face_crop.append(frame[y1-100:y2+100, x1-100:x2+100].copy())  # Crop directly within bounds

                        for crop in face_crop:
                            executor.submit(check_faces, crop)

                cv2.imshow("AI", alpha_frame)
            if cv2.getWindowProperty("AI", cv2.WND_PROP_VISIBLE) < 1:
                break  # Exit the loop when the window is closed


            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
       
       cap.release()
       cv2.destroyAllWindows()
       executor.shutdown(wait=True)

    print(len(faces))
    for idx, crop in enumerate(faces):
        if crop.size != 0:  
            cv2.imshow(f"Crop {idx}", crop)
            key = cv2.waitKey(0)  
            if key == ord('q'):   
                break
    cv2.destroyAllWindows()
    student_atab()


def student_atab():
    for i in range(len(faces)):
        if faces[i] is None or faces[i].size == 0:
            print(f"Face {i} is invalid, skipping...")
            continue

        for j in data:
            main_image = cv2.imread(path_of_dataset + f'/{j}')
            if main_image is None:
                print(f"Failed to load image {j}, skipping...")
                continue

            try:
                result = DeepFace.verify(main_image, faces[i])
                if result['verified']:
                    atab_stu['present'].append(j.split(sep='.')[0])
            except Exception as e:
                print(f"Error verifying face {i} and image {j}: {e}")
                continue  # Just skip this pair, don't restart!
            

            
            

def student_menue():
    global faces
    global atab_stu
    atab_stu['present']=[]
    faces=[]
    face_detector()
    
def resizer(x,mode):
    from design import window_size
    if mode=='x':
        return (x*window_size['x'])/1000
    else:
        return (x*window_size['y'])/1000
def paths(x):
    return os.path.join(os.getcwd(),x)

