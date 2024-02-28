import cv2
import numpy as np
import json
import os
from tqdm import tqdm
import pathlib
import random
import copy
import argparse

boundaries = [
    ([0, 120, 0], [140, 255, 100]), #VERDE Izquierda
    ([25, 0, 75], [180, 38, 255]) #ROJO Derecha
]

N_TRAIN = 30
N_TEST = 10
videoPath = "./pruebas/all"
#videoPathAugmented = "./pruebas/dataAugmented"
nameDataset = 'dataset_colored_segment'
def getClassId(filename):
    infoFile = filename.split('/')[1]
    return infoFile.split('_')[0][1:]

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

def processVideo(video,path,color):
    cap = cv2.VideoCapture(video)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    outFormat = 0
    frameFormat = cv2.COLOR_BGR2GRAY
    if color:
        outFormat = 1
        frameFormat = cv2.COLOR_BGR2RGB
    out = cv2.VideoWriter(path, fourcc, 30.0, (224,224),outFormat)
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
        frame = cv2.cvtColor(frame, frameFormat)
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

def downloadVideos(rootPath,videoPaths,destPath,color):
    print(f"{destPath}:")
    for video in tqdm(videoPaths):
        videoDest = video.replace('mp4','avi')
        processVideo(os.path.join(rootPath,video),str(destPath)+'/'+videoDest,color)

def augmentData(label,nameDataset,color):
    videoPathAugmented = "./pruebas/dataAugmented"
    videoListAug = os.listdir(videoPathAugmented)
    labelAug = copy.deepcopy(label)
    labelsAugVids = getVideosPerClass(labelAug,videoListAug)    
    print('Augmenting data...')
    for videoInfo in labelsAugVids:
        videoNames = videoInfo['videoNames']
        fullPath =  pathlib.Path(f'./{nameDataset}/train/'+videoInfo["name"])
        print(f'Processing Augmented {videoInfo["name"]}...')
        downloadVideos(videoPathAugmented,videoNames,fullPath,color)

def createDataset(qTrain,qTest):
    parser = argparse.ArgumentParser()
    parser.add_argument("--aug", action='store_true', help="Determines if you want to include augmented data")
    parser.add_argument("--name",action="store",help="Determines name of the dataset. Defaults to 'dataset'")
    parser.add_argument("--color",action="store_true",help="Add color to videos")
    args = parser.parse_args()
    if args.name is not None:
        nameDataset = args.name
    videoList = os.listdir(videoPath)
    #videoListAug = os.listdir(videoPathAugmented)
    f = open('./dataset.json')
    label = json.load(f)
    # labelAug = copy.deepcopy(label)
    labelsWithVids = getVideosPerClass(label,videoList)
    # labelsAugVids = getVideosPerClass(labelAug,videoListAug)
    dataset_dir = pathlib.Path(f'./{nameDataset}')
    train_dir = pathlib.Path(f'./{nameDataset}/train')
    test_dir = pathlib.Path(f'./{nameDataset}/test')
    val_dir = pathlib.Path(f'./{nameDataset}/val')
    os.mkdir(dataset_dir)
    os.mkdir(train_dir)
    os.mkdir(test_dir)
    os.mkdir(val_dir)
    for videoInfo in labelsWithVids:
        videoNames = videoInfo['videoNames']
        print(len(videoNames))
        random.shuffle(videoNames)
        trainVideos = videoNames[:qTrain]
        testVideos = videoNames[qTrain:qTrain+qTest]
        valVideos = videoNames[qTrain+qTest:]
        print(f'Processing {videoInfo["name"]}...')

        os.mkdir(os.path.join(train_dir,videoInfo["name"]))
        fullPath =  pathlib.Path(f'./{nameDataset}/train/'+videoInfo["name"])
        downloadVideos(videoPath,trainVideos,fullPath,args.color)

        os.mkdir(os.path.join(test_dir,videoInfo["name"]))
        fullPath =  pathlib.Path(f'./{nameDataset}/test/'+videoInfo["name"])
        downloadVideos(videoPath,testVideos,fullPath,args.color)

        os.mkdir(os.path.join(val_dir,videoInfo["name"]))
        fullPath =  pathlib.Path(f'./{nameDataset}/val/'+videoInfo["name"])
        downloadVideos(videoPath,valVideos,fullPath,args.color)
    if args.aug:
        augmentData(label,nameDataset,args.color)
createDataset(N_TRAIN,N_TEST)