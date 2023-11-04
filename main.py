import cv2
import mediapipe as mp
import numpy as np

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
        black_background[minHeight:maxHeight,left:right] = cv2.cvtColor(hand['img'],cv2.COLOR_RGB2GRAY)
    return black_background

#image = cv2.imread('./mano2.jpg')

cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('prueba1.avi', fourcc, 30.0, (width,height), 0)
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
        cv2.imshow('Hand Canva',handsCanva)
        out.write(handsCanva)
        #if len(hands_img) > 0:
        #    for hand in hands_img:
        #        if (0 not in hand['img'].shape):
        #            cv2.imshow(f"{hand['type']} hand",hand['img']) 
        #cleanHandWindows(hands_img)
        cv2.imshow("Image", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
out.release
cv2.destroyAllWindows()

