import cv2
import os
import pathlib
import re

videoPath = "C:\\Users\\Lucio\\Documents\\all"

videoList = os.listdir(videoPath)

size = 255
def mostrarVideo(video,i):
    videoUrl = str(os.path.join(videoPath,video))
    cap = cv2.VideoCapture(videoUrl)
    live = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(f'sena{i}.avi', fourcc, 30.0, (size,size), 0)
    while cap.isOpened():
        ret,frame = cap.read()
        retLive,frameLive =live.read()
        if ret == False:
            break
        if cv2.waitKey(1) & 0xFF == 27:
            break                
        cv2.imshow('output',frame)
        if retLive:
            cv2.imshow('live',frameLive)
            out.write(frameLive)
    cap.release()
    out.release()


for i in range(1,65):
    videosType = list(filter(lambda x:re.search(f"^{i:03}",x),videoList))
    for j in range(2):
        mostrarVideo(videosType[j],i)
