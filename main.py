import cv2
import mediapipe as mp
import numpy as np
import os
import argparse

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def identifyHand(img,landmark,width,height,label):
    maxHeight = int((max(landmark, key = lambda p: p.y).y)*height)+20
    minHeight = int((min(landmark, key = lambda p:p.y).y)*height)-20
    left =int((min(landmark, key = lambda p:p.x).x) *width)-20
    right = int((max(landmark, key = lambda p:p.x).x)*width)+20
    onlyHands = {
        'img':img[minHeight:maxHeight,left:right],
        'type': label,
        'originalPos':(minHeight,maxHeight,left,right)
    }
    return onlyHands

def getHands(img,width,height,results):
    handImgs = []
    if results.multi_hand_landmarks is not None:
        for i,hand_landmarks in enumerate(results.multi_hand_landmarks):
            label = results.multi_handedness[i].classification[0].label
            handImgs.append(identifyHand(img,hand_landmarks.landmark,width,height,label))
    return handImgs

def cleanHandWindows(hands_img):
    if len(hands_img) == 0:
        if (cv2.getWindowProperty("Right hand",cv2.WND_PROP_VISIBLE) > 0):
            cv2.destroyWindow("Right hand")
        if (cv2.getWindowProperty("Left hand",cv2.WND_PROP_VISIBLE) > 0):
            cv2.destroyWindow("Left hand")
    elif len(hands_img) == 1:
        if hands_img[0]['type'] == 'Right' and cv2.getWindowProperty("Left hand",cv2.WND_PROP_VISIBLE) > 0:
            cv2.destroyWindow("Left hand")
        elif hands_img[0]['type'] == 'Left' and cv2.getWindowProperty("Right hand",cv2.WND_PROP_VISIBLE) > 0:
            cv2.destroyWindow("Right hand")   
    else:
        return
    
def createHandsCanva(height,width,hands_img):
    black_background = np.zeros((height,width),dtype = np.uint8)
    for hand in hands_img:
        minHeight,maxHeight,left,right = hand['originalPos']
        if len(hand['img']) != 0 and maxHeight-minHeight >= hand['img'].shape[0] and right-left >= hand['img'].shape[1] and not 0 in hand['img'].shape:
            #print("Alto x Ancho:",hand['img'].shape)
            #print("Pos x:",(right,left))
            #print("Pos y:",(maxHeight,minHeight))
            black_background[minHeight:maxHeight,left:right] = cv2.cvtColor(hand['img'],cv2.COLOR_RGB2GRAY)
    return black_background

def resizeFrame(frameCanva,frameHand,width,height,ratioHand):
    heightHand = frameHand.shape[0]
    widthtHand = frameHand.shape[1]
    supHand = heightHand*widthtHand
    supCanva = width*height
    reduceRatio =  (ratioHand*supCanva)/supHand
    print(reduceRatio)
    newWidth = int(width*reduceRatio)
    newHeight = int(height*reduceRatio)
    frameCanva = cv2.resize(frameCanva,(newWidth,newHeight))
    #frameCanva = cv2.rectangle(frameCanva,(0,0),(newWidth,newHeight),(255,0,0),5,0)
    newCanva = np.zeros((height,width),dtype = np.uint8)
    yoff = round((height-newHeight)/2)
    xoff = round((width-newWidth)/2)
    newCanva[yoff:yoff+newHeight, xoff:xoff+newWidth] = frameCanva
    return newCanva

def main(path = 0, out = 'pruebas/prueba1.avi'):
    cap = cv2.VideoCapture(path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    HEIGHT_CANVA = height
    WIDTH_CANVA = width
    RATIO_HAND = 0.02
    print(f'Path del video:{path}\nMedidas del video:{width}x{height}')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    print(f'Lo escribo en {out}')
    out = cv2.VideoWriter(out, fourcc, 30.0, (WIDTH_CANVA,HEIGHT_CANVA), 0)
    with mp_hands.Hands(
            static_image_mode= False,
            max_num_hands= 2,
            min_detection_confidence= 0.8,
            min_tracking_confidence = 0.5
        ) as hands:
        while cap.isOpened():
            ret,frame = cap.read()
            if ret == False:
                break
            height,width,_ = frame.shape
            frame = cv2.flip(frame,1)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            hands_img = getHands(frame,width,height,results)
            handsCanva = createHandsCanva(height,width,hands_img)
            supOr = width*height
            if len(hands_img) > 0:
                supHand = hands_img[0]['img'].shape[0]*hands_img[0]['img'].shape[1]
                print("Superficie mano/frame",supHand/supOr)
            #if len(hands_img) > 0 and supHand/supOr > RATIO_HAND:
                #handsCanva = resizeFrame(handsCanva,hands_img[0]['img'],width,height,RATIO_HAND)
            cv2.imshow('Hand Canva',handsCanva)
            #if (len(hands_img)>0):
            out.write(handsCanva) 
            cleanHandWindows(hands_img)
            cv2.imshow("Image", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

parser = argparse.ArgumentParser()

parser.add_argument("--live", action='store_true', help="Determines if its input its a live video stream")
args = parser.parse_args()

if args.live:
    main()
else:
    input_path = './pruebas/serie_prueba_sinGuantes'
    videoPaths = os.listdir(input_path)
    #videoPaths = videoPaths[45:]
    for index,video_path in enumerate(videoPaths):
        output_path = f'./pruebas/processedVids/sign{index+1:02}.avi'
        main(path=input_path+'/'+video_path,out=output_path)