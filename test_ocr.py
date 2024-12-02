from paddleocr import PaddleOCR

# Also switch the language by modifying the lang parameter
ocr = PaddleOCR(lang="en") # The model file will be downloaded automatically when executed for the first time
img_path ='./license_plate_crop.jpg'
result = ocr.ocr(img_path)
for line in result[0]:
    print(line[1][0])
