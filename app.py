import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import cv2
import easyocr
import pandas as pd
from util import set_background, write_csv
import uuid
import os
from  streamlit_webrtc import webrtc_streamer
import av
from transformers import AutoModel, AutoTokenizer

set_background("./imgs/background.png")


tokenizer = AutoTokenizer.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True)
model = AutoModel.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True, low_cpu_mem_usage=True, device_map='cpu', use_safetensors=True, pad_token_id=tokenizer.eos_token_id)
model = model.eval().cuda()

folder_path = "./licenses_plates_imgs_detected/"
LICENSE_MODEL_DETECTION_DIR = './models/results/runs/detect/train3/weights/best.onnx'
COCO_MODEL_DIR = "./models/results/yolo11n.pt"

reader = easyocr.Reader(['en'], gpu=False)

vehicles = [2]

header = st.container()
body = st.container()

# NOTE: run model!
coco_model = YOLO(COCO_MODEL_DIR)
license_plate_detector = YOLO(LICENSE_MODEL_DETECTION_DIR)

threshold = 0.15

state = "Uploader"

if "state" not in st.session_state :
    st.session_state["state"] = "Uploader"

class VideoProcessor:
    def recv(self, frame) :
        img = frame.to_ndarray(format="bgr24")
        img_to_an = img.copy()
        img_to_an = cv2.cvtColor(img_to_an, cv2.COLOR_RGB2BGR)
        license_detections = license_plate_detector(img_to_an)[0]

        if len(license_detections.boxes.cls.tolist()) != 0 :
            for license_plate in license_detections.boxes.data.tolist() :
                x1, y1, x2, y2, score, class_id = license_plate

                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)

                license_plate_crop = img[int(y1):int(y2), int(x1): int(x2), :]
            
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY) 

                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_gray, img)

                cv2.rectangle(img, (int(x1) - 40, int(y1) - 40), (int(x2) + 40, int(y1)), (255, 255, 255), cv2.FILLED)
                cv2.putText(img,
                            str(license_plate_text),
                            (int((int(x1) + int(x2)) / 2) - 70, int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 0),
                            3)

        return av.VideoFrame.from_ndarray(img, format="bgr24")
    


# NOTE: detect id vehicles
def model_prediction(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    object_detections = coco_model(img)[0]
    
    license_detections = license_plate_detector(img)[0]

    if len(object_detections.boxes.cls.tolist()) != 0 :

        # NOTE: set variable model detect
        for detection in object_detections.boxes.data.tolist() :
            xcar1, ycar1, xcar2, ycar2, car_score, class_id = detection

            if int(class_id) in vehicles :
                cv2.rectangle(img, (int(xcar1), int(ycar1)), (int(xcar2), int(ycar2)), (0, 0, 255), 3)
    else :
            xcar1, ycar1, xcar2, ycar2 = 0, 0, 0, 0
            car_score = 0



    if len(license_detections.boxes.cls.tolist()) != 0 :
        license_plate_crops_total = []

        for license_plate in license_detections.boxes.data.tolist() :
            x1, y1, x2, y2, score, class_id = license_plate

            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)

            license_plate_crop = img[int(y1):int(y2), int(x1): int(x2), :]

            # Convert and save the cropped license plate image
            license_plate_crop_rgb = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2RGB)

            # Save the image to a specific path
            output_path = "license_plate_crop.jpg"  # Update the path as needed
            cv2.imwrite(output_path, cv2.cvtColor(license_plate_crop_rgb, cv2.COLOR_RGB2BGR))
            
            res = model.chat(tokenizer, './license_plate_crop.jpg', ocr_type='ocr')
            print(res)
            
            return {'License_plate_crop': license_plate_crop_rgb, 'License_id': res}
    
    else : 
        img_wth_box = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return {'License_plate_crop': img_wth_box, 'License_id': None}
    

def change_state_uploader() :
    st.session_state["state"] = "Uploader"

    
def change_state_camera() :
    st.session_state["state"] = "Camera"

def change_state_live() :
    st.session_state["state"] = "Live"
    
with header :
    _, col1, _ = st.columns([0.2,1,0.1])
    col1.title("💥 License Car Plate Detection 🚗")

    _, col0, _ = st.columns([0.15,1,0.1])
    col0.image("./imgs/test_background.jpg", width=500)


    _, col4, _ = st.columns([0.1,1,0.2])
    col4.subheader("Computer Vision Detection with YoloV8 🧪")

    _, col, _ = st.columns([0.3,1,0.1])
    col.image("./imgs/plate_test.jpg")

    _, col5, _ = st.columns([0.05,1,0.1])

    st.write("The differents models detect the car and the license plate in a given image, then extracts the info about the license using EasyOCR, and crop and save the license plate as a Image, with a CSV file with all the data.   ")


with body :
    _, col1, _ = st.columns([0.1,1,0.2])
    col1.subheader("Check It-out the License Car Plate Detection Model 🔎!")

    _, colb1, colb2, colb3 = st.columns([0.2, 0.7, 0.6, 1])

    if colb1.button("Upload an Image", on_click=change_state_uploader) :
        pass
    elif colb2.button("Take a Photo", on_click=change_state_camera) :
        pass
    elif colb3.button("Live Detection", on_click=change_state_live) :
        pass

    if st.session_state["state"] == "Uploader" :
        img = st.file_uploader("Upload a Car Image: ", type=["png", "jpg", "jpeg"])
    elif st.session_state["state"] == "Camera" :
        img = st.camera_input("Take a Photo: ")
    elif st.session_state["state"] == "Live" :
        webrtc_streamer(key="sample", video_processor_factory=VideoProcessor)
        img = None

    _, col2, _ = st.columns([0.3,1,0.2])

    _, col5, _ = st.columns([0.8,1,0.2])

    
    if img is not None:
        image = np.array(Image.open(img))    
        col2.image(image)

        if col5.button("Apply Detection"):
            results = model_prediction(image)
                




 
