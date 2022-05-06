"""from kivy.graphics.texture import Texture
from kivymd.app import MDApp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDRaisedButton
from kivy.uix.image import Image
from kivy.clock import Clock
import cv2


class MainApp(MDApp):

    def build(self):
        layout = MDBoxLayout(orientation='vertical')
        self.image = Image()
        layout.add_widget(self.image)
        layout.add_widget(MDRaisedButton(
            text="CLICK",
            pos_hint={'center_x': .5, 'center_y': .5},
            size_hint=(None, None))
        )
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.load_video, 1.0 / 30.0)
        return layout

    def load_video(self, *args):
        ret, frame = self.capture.read()
        self.image_frame = frame
        buffer = cv2.flip(frame, 0).tostring()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
        self.image.texture = texture


if __name__ == '__main__':
    MainApp().run()

import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr

img = cv2.imread('image4.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#print(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
edged = cv2.Canny(bfilter, 30, 200)
keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
location = None
for contour in contours:
    approx = cv2.approxPolyDP(contour, 10, True)
    if len(approx) == 4:
        location = approx
        break
print(location)

mask = np.zeros(gray.shape, np.uint8)
new_image = cv2.drawContours(mask, [location], 0,255, -1)
new_image = cv2.bitwise_and(img, img, mask=mask)

(x,y) = np.where(mask==255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))
cropped_image = gray[x1:x2+1, y1:y2+1]

reader = easyocr.Reader(['en'])
result = reader.readtext(cropped_image)
print(result)
text = result[0][-2]
print(text)"""

from PIL.Image import ImageTransformHandler
import cv2
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe"

cascade = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")
states = {"AN": "Andaman and Nicobar",
          "AP": "Andhra Pradesh", "AR": "Arunachal Pradesh",
          "AS": "Assam", "BR": "Bihar", "CH": "Chandigarh",
          "DN": "Dadra and Nagar Haveli", "DD": "Daman and Diu",
          "DL": "Delhi", "GA": "Goa", "GJ": "Gujarat",
          "HR": "Haryana", "HP": "Himachal Pradesh",
          "JK": "Jammu and Kashmir", "KA": "Karnataka", "KL": "Kerala",
          "LD": "Lakshadweep", "MP": "Madhya Pradesh", "MH": "Maharashtra", "MN": "Manipur",
          "ML": "Meghalaya", "MZ": "Mizoram", "NL": "Nagaland", "OD": "Odissa",
          "PY": "Pondicherry", "PN": "Punjab", "RJ": "Rajasthan", "SK": "Sikkim", "TN": "TamilNadu",
          "TR": "Tripura", "UP": "Uttar Pradesh", "WB": "West Bengal", "CG": "Chhattisgarh",
          "TS": "Telangana", "JH": "Jharkhand", "UK": "Uttarakhand"}


def extract_num(img_filename):
    img = cv2.imread(img_filename)
    # Img To Gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    nplate = cascade.detectMultiScale(image=gray)  # (gray, 1.1, 4)
    # crop portion
    for (x, y, w, h) in nplate:
        wT, hT, cT = img.shape
        a, b = (int(0.02 * wT), int(0.02 * hT))
        plate = img[y + a:y + h - a, x + b:x + w - b, :]
        # make the img more darker to identify LPR
        kernel = np.ones((1, 1), np.uint8)
        plate = cv2.dilate(plate, kernel, iterations=1)
        plate = cv2.erode(plate, kernel, iterations=1)
        plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        (thresh, plate) = cv2.threshold(plate_gray, 127, 255, cv2.THRESH_BINARY)
        # read the text on the plate
        read = pytesseract.image_to_string(plate)
        read = ''.join(e for e in read if e.isalnum())
        stat = read[0:2]

        cv2.rectangle(img, (x, y), (x + w, y + h), (51, 51, 255), 2)
        cv2.rectangle(img, (x - 1, y - 40), (x + w + 1, y), (51, 51, 255), -1)
        cv2.putText(img, read, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.imshow("plate", plate)

    cv2.imwrite("Result.png", img)
    cv2.imshow("Result", img)
    #print(read)
    if cv2.waitKey(0) == 113:
        exit()
    cv2.destroyAllWindows()


extract_num("drc5.jpg")
"""
import subprocess

from PIL.Image import ImageTransformHandler
import cv2
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = "D:/TFBapk/mobile/tesseract-ocr-w64-setup-v5.0.1.20220118.exe"


cascade = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")
states = {"AN": "Andaman and Nicobar",
          "AP": "Andhra Pradesh", "AR": "Arunachal Pradesh",
          "AS": "Assam", "BR": "Bihar", "CH": "Chandigarh",
          "DN": "Dadra and Nagar Haveli", "DD": "Daman and Diu",
          "DL": "Delhi", "GA": "Goa", "GJ": "Gujarat",
          "HR": "Haryana", "HP": "Himachal Pradesh",
          "JK": "Jammu and Kashmir", "KA": "Karnataka", "KL": "Kerala",
          "LD": "Lakshadweep", "MP": "Madhya Pradesh", "MH": "Maharashtra", "MN": "Manipur",
          "ML": "Meghalaya", "MZ": "Mizoram", "NL": "Nagaland", "OD": "Odissa",
          "PY": "Pondicherry", "PN": "Punjab", "RJ": "Rajasthan", "SK": "Sikkim", "TN": "TamilNadu",
          "TR": "Tripura", "UP": "Uttar Pradesh", "WB": "West Bengal", "CG": "Chhattisgarh",
          "TS": "Telangana", "JH": "Jharkhand", "UK": "Uttarakhand"}


def extract_num(img_filename):
    img = cv2.imread(img_filename)
    # Img To Gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    nplate = cascade.detectMultiScale(gray, 1.1, 4)
    # crop portion
    for (x, y, w, h) in nplate:
        wT, hT, cT = img.shape
        a, b = (int(0.02 * wT), int(0.02 * hT))
        plate = img[y + a:y + h - a, x + b:x + w - b, :]
        # make the img more darker to identify LPR
        kernel = np.ones((1, 1), np.uint8)
        plate = cv2.dilate(plate, kernel, iterations=1)
        plate = cv2.erode(plate, kernel, iterations=1)
        plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        (thresh, plate) = cv2.threshold(plate_gray, 127, 255, cv2.THRESH_BINARY)
        # read the text on the plate
        read = pytesseract.image_to_string(plate)
        read = ''.join(e for e in read if e.isalnum())
        stat = read[0:2]
        cv2.rectangle(img, (x, y), (x + w, y + h), (51, 51, 255), 2)
        cv2.rectangle(img, (x - 1, y - 40), (x + w + 1, y), (51, 51, 255), -1)
        cv2.putText(img, read, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.imshow("plate", plate)

    cv2.imwrite("Result.png", img)
    cv2.imshow("Result", img)
    if cv2.waitKey(0) == 113:
        exit()
    cv2.destroyAllWindows()


extract_num("car_img.png")
"""
