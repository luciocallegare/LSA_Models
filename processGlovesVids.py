import cv2
import numpy as np
import os
from tqdm import tqdm
import pathlib

boundaries = [
    ([0, 120, 0], [140, 255, 100]), #VERDE Izquierda
    ([25, 0, 75], [180, 38, 255]) #ROJO Derecha
]


inputPath = pathlib.Path("./pruebas/serie_prueba_guantes")
outputPath = pathlib.Path("./pruebas/processedVidsGloves")

def handSegment(frame):
    lower, upper = boundaries[0]
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")
    mask1 = cv2.inRange(frame, lower, upper)

    lower, upper = boundaries[1]
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")
    mask2 = cv2.inRange(frame, lower, upper)
    mask = cv2.bitwise_or(mask1, mask2)
    output = cv2.bitwise_and(frame, frame, mask=mask)
    return output

def handSegmentByHand(frame,hand):
    lower, upper = boundaries[0] if hand == 'L' else boundaries[1]
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")
    mask = cv2.inRange(frame, lower, upper)
    output = cv2.bitwise_and(frame, frame, mask=mask)
    return output

def processVideo(video,path):
    cap = cv2.VideoCapture(video)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(path, fourcc, 30.0, (224,224),0)
    nframe = 0
    #print('PROCESS VIDEO PATH',path)
    #print('PROCESS VIDEO VIDEO',video)
    while cap.isOpened():
        ret,frame = cap.read()
        if ret == False:
            break
        nframe += 1
        frame = cv2.resize(frame,(224,224))        
        frame = cv2.flip(frame,1)
        frame = handSegment(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #cv2.imshow('output',frame)
        #if cv2.waitKey(1) & 0xFF == 27:
            #break
        out.write(frame)
    cap.release()
    out.release()

def getVideosPerClass(dict,videos):
    for elem in dict:
        idVids ='0' + elem['id']
        elem['videoNames'] = list(filter(lambda x:x.split('_')[0] == idVids,videos))
    return dict

def downloadVideos(rootPath,videoPaths,destPath):
    print(f"{destPath}:")
    for video in tqdm(videoPaths):
        videoDest = video.replace('mp4','avi')
        processVideo(os.path.join(rootPath,video),str(destPath)+'/'+videoDest)

if not os.path.isdir(outputPath):
    os.mkdir(outputPath)
listVids = os.listdir(inputPath)

downloadVideos(inputPath,listVids,outputPath)