import cv2

videoPath = './pruebas/serie_prueba_sinGuantes1.mp4'

cap = cv2.VideoCapture(videoPath)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('./pruebas/serie_prueba_sinGuantes1_60fps.avi', fourcc, 60.0, (848,480),3)

while cap.isOpened():
    ret,frame = cap.read()
    if ret == False:
        break
    out.write(frame)
    out.write(frame)
cap.release()
out.release()