import cv2
import numpy as np
import dlib
from math import hypot

URL = "http://192.168.100.7:8080/video"
def midpoint(p1,p2):
    return int((p1.x +p2.x)/2) , int((p1.y + p2.y)/2)

#Load image of sticker
sun = cv2.imread("D:\\Vinhi\\cyclone.png")
#Load videocapture
cap = cv2.VideoCapture(URL)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('D:\\Data_Detection_csv\\shape_predictor_68_face_landmarks.dat')
while True:
   _, frame = cap.read()
   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   faces = detector(gray)
   for face in faces:
      x, y = face.left() , face.top()
      x1, y1 = face.right(), face.bottom()
      landmarks = predictor(gray, face)
      left_pt = (landmarks.part(36).x,landmarks.part(36).y)
      right_pt =(landmarks.part(39).x, landmarks.part(39).y)
      sun_width = int(hypot(left_pt[0]-right_pt[0],left_pt[1]-right_pt[1])*1.25)
      sun_height = int(sun_width*.77)
# Sticker pos
      p11 = (landmarks.part(1).x,landmarks.part(1).y)
      p15 = (landmarks.part(15).x, landmarks.part(15).y)
      p28 = (landmarks.part(28).x,landmarks.part(28).y)
      p29 = (landmarks.part(29).x, landmarks.part(29).y)
      pm1 = midpoint(landmarks.part(28),landmarks.part(29))
      pm2 = ((p11[0] + pm1[0])/2,(p11[1] + pm1[1])/2)
      pm3 = ((p15[0] + pm1[0])/2,(p15[1] + pm1[1])/2)
      top_left1 = (int(pm2[0] - sun_width/2), int(pm2[1] - sun_height/2))
      bot_right1 = (int(pm2[0] + sun_width/2), int(pm2[1] + sun_height/2))
      top_left2 = (int(pm3[0] - sun_width / 2), int(pm3[1] - sun_height / 2))
      bot_right2 = (int(pm3[0] + sun_width / 2), int(pm3[1] + sun_height / 2))
        #Adding sticker
      sun_img = cv2.resize(sun,(sun_width, sun_height))
      sun_img_gray = cv2.cvtColor(sun_img, cv2.COLOR_BGR2GRAY)
      _, sun_mask = cv2.threshold(sun_img_gray, 25, 255, cv2.THRESH_BINARY_INV)
      sun_area1 = frame[top_left1[1]: top_left1[1] + sun_height,
                         top_left1[0]: top_left1[0] + sun_width]
      sun_area2 = frame[top_left2[1]: top_left2[1] + sun_height,
                        top_left2[0]: top_left2[0] + sun_width]
      sun_area_no_sun1 = cv2.bitwise_and(sun_area1,sun_area1,mask=sun_mask)
      sun_area_no_sun2 = cv2.bitwise_and(sun_area2,sun_area2, mask=sun_mask)
      final_sun1 = cv2.add(sun_area_no_sun1,sun_img)
      final_sun2 = cv2.add(sun_area_no_sun2,sun_img)
      frame[top_left1[1]: top_left1[1] + sun_height,
                top_left1[0]: top_left1[0] + sun_width] = final_sun1
      frame[top_left2[1]: top_left2[1] + sun_height,
                top_left2[0]: top_left2[0] + sun_width] = final_sun2
   if frame is not None:
      cv2.imshow("Frame", frame)
   key = cv2.waitKey(1)
   if key == 27:
      break
cap.release()
cv2.destroyAllWindows()