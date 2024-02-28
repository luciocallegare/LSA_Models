import os
import cv2
import numpy as np
import vidaug.augmentors as va
import random

pathDataset = './pruebas/all'
pathAug = './pruebas/dataAugmented'

videoList = os.listdir(pathDataset)

def divide_chunks(l, n): 
      
    # looping till length l 
    for i in range(0, len(l), n):  
        yield l[i:i + n] 

videoListDivided = list(divide_chunks(videoList,50))

vids = []

for vidList in videoListDivided:
    random.shuffle(vidList)
    vids += vidList[:15]

for video in vids:
    typeVid = video.split('_')[0]
    input_video_path = f'{pathDataset}/{video}'
    nombre, extension = os.path.splitext(input_video_path)
    nombre = nombre.split('/')[3]
    output_video_path = f'{pathAug}/{nombre}_aug{extension}'
    # Load the video file
    cap = cv2.VideoCapture(input_video_path)

    # Define the output video writer
    # Meta.    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (width, height)    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # Video writer.
    out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)
    # collect all frames of the video
    frames = []
    print(f'Tratando {input_video_path}')
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frames.append(frame)
        else:
            break
    cap.release()
    cropWidth = int(width/2)
    cropHeight = int(height/2)
    # Apply the video augmentation pipeline to each frame of the video
    sometimes = lambda aug: va.Sometimes(0.5, aug) # Used to apply augmentor with 50% probability
    seq = va.Sequential([
        sometimes(va.RandomTranslate(100,250)),
        va.CenterCrop((cropHeight+100,cropWidth+100)),
    ])
    #augment the frames
    video_aug = seq(frames)
    
    # output the video
    for frame in video_aug:
        frame = cv2.resize(frame,frame_size)
        out.write(frame)
    out.release()