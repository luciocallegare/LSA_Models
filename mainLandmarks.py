import cv2
import mediapipe as mp
import numpy as np
import os
import argparse
from keras.models import load_model
import json
import time

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
containers = mp.tasks.components.containers
MEASURES_MODEL = 224
N_FRAMES = 20

def createHandsCanva(height,width,results):
    black_background = np.zeros((height,width,3),dtype = np.uint8)
    for i,hand_landmarks in enumerate(results.multi_hand_landmarks):
        mp_drawing.draw_landmarks(black_background,
                                    hand_landmarks,
                                    mp_hands.HAND_CONNECTIONS,
                                    #mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                                    #mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=10, circle_radius=2)
                                    )
    return black_background

def main(path = 0, out = 'pruebas/prueba1.avi'):
    cap = cv2.VideoCapture(path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    HEIGHT_CANVA = height
    WIDTH_CANVA = width
    #print(f'Path del video:{path}\nMedidas del video:{width}x{height}')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(out, fourcc, 30.0, (WIDTH_CANVA,HEIGHT_CANVA), 1)
    f = open('./dataset.json')
    label = json.load(f)
    threshold = 0.5
    model = load_model('./models/modelConvLSTM', compile=False)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    with mp_hands.Hands(
            static_image_mode= False,
            max_num_hands= 2,
            min_detection_confidence= 0.8,
            min_tracking_confidence = 0.5
        ) as hands:
        sequence = []
        time_stamp_before = time.time()
        i = 0
        while cap.isOpened():
            ret,frame = cap.read()
            if ret == False:
                break
            time_stamp_after = time.time()
            fps = 1/(time_stamp_after-time_stamp_before)
            video_frames_count = fps*2
            skip_frames_window = int(video_frames_count/N_FRAMES)
            frame = cv2.flip(frame,1)
            cv2.imshow("Image", frame)
            if i < skip_frames_window:
                i+=1
                continue
            time_stamp_before = time_stamp_after
            height,width,_ = frame.shape
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            if results.multi_hand_landmarks is not None:
                handsCanva = createHandsCanva(HEIGHT_CANVA,WIDTH_CANVA,results)
                handsCanva = cv2.cvtColor(handsCanva, cv2.COLOR_BGR2GRAY)
                handsCanva =  cv2.resize(handsCanva,(MEASURES_MODEL,MEASURES_MODEL))
                cv2.imshow('Hand Canva',handsCanva)
                sequence.append(handsCanva)
                out.write(handsCanva) 
            else:
                i = 0
                sequence = []
            if len(sequence) == N_FRAMES:
                cantHands = len(results.multi_hand_landmarks)
                probArray =  model.predict(np.expand_dims(sequence,0))
                idGesture =  np.argmax(probArray)
                probPred = probArray[0][idGesture]
                cantHandsPred = 1 if label[idGesture]['handsUsed'] != 'B' else 2
                print(idGesture,probPred)
                if probPred > threshold and cantHands == cantHandsPred:
                    print(label[idGesture]["name"])
                sequence = []
            if cv2.waitKey(1) & 0xFF == 27:
                break
            i = 0

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
        output_path = f'./pruebas/landmarksVids/sign{index+1:02}.avi'
        main(path=input_path+'/'+video_path,out=output_path)