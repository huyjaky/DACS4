import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import cv2
import easyocr
from util import set_background
from transformers import AutoModel, AutoTokenizer
from paddleocr import PaddleOCR

set_background("./imgs/background.png")

ocr = PaddleOCR(
    lang="en"
)  # The model file will be downloaded automatically when executed for the first time

tokenizer = AutoTokenizer.from_pretrained("ucaslcl/GOT-OCR2_0", trust_remote_code=True)
ocr_model = AutoModel.from_pretrained(
    "ucaslcl/GOT-OCR2_0",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    device_map="cuda",
    use_safetensors=True,
    pad_token_id=tokenizer.eos_token_id,
)
ocr_model = ocr_model.eval().cuda()

detect_car_model = YOLO(
    "/home/duckq1u/Documents/DoAnChuyenNganh/DACN1ver1/models/yolov8n.pt"
)
license_plate_model = YOLO(
    "/home/duckq1u/Documents/DoAnChuyenNganh/DACN1ver1/models/model_ver2/best.pt"
)
reader = easyocr.Reader(["en"], gpu=True)
vehicles = [2]


header = st.container()
body = st.container()


threshold = 0.15

state = "Uploader"

if "state" not in st.session_state:
    st.session_state["state"] = "Uploader"


# NOTE: detect id vehicles
def model_prediction(img):
    license_plate_list = []
    car_list = []
    license_number_list_paddle = []
    license_number_list_got = []
    car_detect = detect_car_model(img)[0]

    if car_detect.boxes.cls.tolist() != 0:
        for car in car_detect.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = car

            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)

            car_crop = img[int(y1) : int(y2), int(x1) : int(x2), :]

            car_list.append(car_crop)

            license_plates = license_plate_model(car_crop)[0]
            if license_plates.boxes.cls.tolist() != 0:
                for license_plate in license_plates.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = license_plate

                    cv2.rectangle(
                        car_crop, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3
                    )

                    license_plate_crop = car_crop[
                        int(y1) : int(y2), int(x1) : int(x2), :
                    ]

                    # Convert and save the cropped license plate image
                    license_plate_crop_rgb = cv2.cvtColor(
                        license_plate_crop, cv2.COLOR_BGR2RGB
                    )

                    # Save the image to a specific path
                    output_path = "license_plate_crop.jpg"  # Update the path as needed
                    cv2.imwrite(
                        output_path,
                        cv2.cvtColor(license_plate_crop_rgb, cv2.COLOR_RGB2BGR),
                    )
                    license_plate_list.append(license_plate_crop)

                    result = ocr.ocr(license_plate_crop)
                    text = ""
                    if result[0] is not None:
                        for line in result[0]:
                            text += line[1][0]
                    license_number_list_paddle.append(text)
                    res = ocr_model.chat(
                        tokenizer, "./license_plate_crop.jpg", ocr_type="ocr"
                    )
                    license_number_list_got.append(res)

    return {
        "car_crop": car_list,
        "license_crop": license_plate_list,
        "license_number": {
            "paddleocr": license_number_list_paddle,
            "gotocr": license_number_list_got,
        },
    }


def change_state_uploader():
    st.session_state["state"] = "Uploader"


def change_state_camera():
    st.session_state["state"] = "Camera"


def change_state_live():
    st.session_state["state"] = "Live"


with header:
    _, col1, _ = st.columns([0.2, 1, 0.1])
    col1.title("💥 License Car Plate Detection 🚗")

    _, col0, _ = st.columns([0.15, 1, 0.1])
    col0.image("./imgs/test_background.jpg", width=500)

    _, col4, _ = st.columns([0.1, 1, 0.2])
    col4.subheader("Computer Vision Detection with YoloV8 🧪")

    _, col, _ = st.columns([0.3, 1, 0.1])
    col.image("./imgs/plate_test.jpg")

    _, col5, _ = st.columns([0.05, 1, 0.1])

    st.write(
        "The differents models detect the car and the license plate in a given image, then extracts the info about the license using EasyOCR, and crop and save the license plate as a Image, with a CSV file with all the data.   "
    )


with body:
    _, col1, _ = st.columns([0.1, 1, 0.2])
    col1.subheader("Check It-out the License Car Plate Detection Model 🔎!")

    _, colb1, colb2, colb3 = st.columns([0.2, 0.7, 0.6, 1])

    if st.session_state["state"] == "Uploader":
        img = st.file_uploader("Upload a Car Image: ", type=["png", "jpg", "jpeg"])
    elif st.session_state["state"] == "Camera":
        img = st.camera_input("Take a Photo: ")
    elif st.session_state["state"] == "Live":
        # webrtc_streamer(key="sample", video_processor_factory=VideoProcessor)
        img = None

    _, col2, _ = st.columns([0.3, 1, 0.2])

    _, col5, _ = st.columns([0.8, 1, 0.2])

    if img is not None:
        image = np.array(Image.open(img))
        col2.image(image)
        results = model_prediction(image)

        image_zone, model_prediction_zone = st.columns(2)

        with image_zone:
            st.image(results["car_crop"])
        with model_prediction_zone:
            st.image(results["license_crop"])
            st.title(results["license_number"]["paddleocr"])
            st.title(results["license_number"]["gotocr"])
