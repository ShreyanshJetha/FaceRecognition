import cv2
import numpy as np
import face_recognition

imgBill=face_recognition.load_image_file('images/shreyansh2.jpg')
imgBill=cv2.cvtColor(imgBill, cv2.COLOR_BGR2RGB)
imgTest=face_recognition.load_image_file('images/shreyansh jetha.jpg')
imgTest=cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

faceLoc=face_recognition.face_locations(imgBill)[0]
encodebill=face_recognition.face_encodings(imgBill)[0]
cv2.rectangle(imgBill, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

faceLoctest=face_recognition.face_locations(imgTest)[0]
encodebilltest=face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLoctest[3], faceLoctest[0]), (faceLoctest[1], faceLoctest[2]), (255, 0, 255), 2)

results=face_recognition.compare_faces([encodebill],encodebilltest)
facedis=face_recognition.face_distance([encodebill],encodebilltest)
cv2.putText(imgTest,f'{results}{round(facedis[0],2)}',(50,50),cv2.FONT_ITALIC,1,(0,0,255),2)

print(results,facedis)
cv2.imshow('Bill gates ', imgBill)
cv2.imshow('Bill gates Test', imgTest)
cv2.waitKey(0)