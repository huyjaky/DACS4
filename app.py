import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import cv2
from util import set_background
from paddleocr import PaddleOCR

set_background("./imgs/background.png")

folder_path = "./licenses_plates_imgs_detected/"
LICENSE_MODEL_DETECTION_DIR = "./models/results/best.onnx"
COCO_MODEL_DIR = "./models/results/yolo11n.pt"

# Also switch the language by modifying the lang parameter
ocr = PaddleOCR(lang="en") # The model file will be downloaded automatically when executed for the first time

vehicles = [2]

header = st.container()
body = st.container()

coco_model = YOLO(COCO_MODEL_DIR)
license_plate_detector = YOLO(LICENSE_MODEL_DETECTION_DIR)


threshold = 0.15

state = "Uploader"

if "state" not in st.session_state:
    st.session_state["state"] = "Uploader"


def model_prediction(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    license_detections = license_plate_detector(img)[0]

    if len(license_detections.boxes.cls.tolist()) != 0:
        for license_plate in license_detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)

            license_plate_crop = img[int(y1) : int(y2), int(x1) : int(x2), :]

            # Convert and save the cropped license plate image
            license_plate_crop_rgb = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2RGB)

            # Save the image to a specific path
            output_path = "license_plate_crop.jpg"  # Update the path as needed
            cv2.imwrite(
                output_path, cv2.cvtColor(license_plate_crop_rgb, cv2.COLOR_RGB2BGR)
            )

        img_path ='./license_plate_crop.jpg'
        result = ocr.ocr(img_path)
        text = ''
        for line in result[0]:
            text += line[1][0]

        return {
            "image": "./license_plate_crop.jpg",
            "text": text
        }

    else:
        img_wth_box = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return {"image": img_wth_box, "text": ""}


def change_state_uploader():
    st.session_state["state"] = "Uploader"


def change_state_camera():
    st.session_state["state"] = "Camera"


def change_state_live():
    st.session_state["state"] = "Live"


with header:
    _, col1, _ = st.columns([0.2, 1, 0.1])
    col1.title("💥 License Car Plate Detection 🚗")

    _, col4, _ = st.columns([0.1, 1, 0.2])
    col4.subheader("Computer Vision Detection with YoloV8 🧪")

    _, col, _ = st.columns([0.3, 1, 0.1])
    col.image("./imgs/plate_test.jpg")

    _, col5, _ = st.columns([0.05, 1, 0.1])

    st.write(
        "The differents models detect the car and the license plate in a given image, then extracts the info about the license using EasyOCR, and crop and save the license plate as a Image, with a CSV file with all the data.   "
    )


with body:
    if st.session_state["state"] == "Uploader":
        img = st.file_uploader("Upload a Car Image: ", type=["png", "jpg", "jpeg"])

    _, col2, _ = st.columns([0.3, 1, 0.2])

    _, col5, _ = st.columns([0.8, 1, 0.2])

    if img is not None:
        image = np.array(Image.open(img))
        col2.image(image, width=400)

        image_zone, model_prediction_zone = st.columns(2)

        result = model_prediction(image)

        with image_zone:
            st.image(result["image"])
        with model_prediction_zone:
            st.title(result["text"])
