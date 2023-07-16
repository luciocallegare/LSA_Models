import cv2
import mediapipe as mp
import math

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

def identifyHands(landmark,frame,width,height,i):
    maxHeight = int((max(landmark, key = lambda p: p.y).y)*height)+20
    minHeight =int((min(landmark, key = lambda p:p.y).y)*height)-20
    left =int((min(landmark, key = lambda p:p.x).x) *width)-20
    right = int((max(landmark, key = lambda p:p.x).x)*width)+20
    onlyHands = frame[minHeight:maxHeight,left:right]
    if 0 not in onlyHands.shape:
        cv2.imshow(f'{i} hand',onlyHands)
    return cv2.rectangle(frame,(left,maxHeight),(right,minHeight),(0,0,255),4)

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
        frame_rgb = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks is not None:
            for i,hand_landmarks in enumerate(results.multi_hand_landmarks):
                label = results.multi_handedness[i].classification[0].label
                mp_drawing.draw_landmarks( frame,hand_landmarks,mp_hands.HAND_CONNECTIONS)
                frame = identifyHands(hand_landmarks.landmark,frame,width,height,label)
        cv2.imshow("Frame",frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
    