import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture('./prueba.mp4')
COLOR_LHAND = np.array([59,169,54], dtype='uint8') #VERDE
COLOR_RHAND = np.array((133,36,59), dtype='uint8') #ROJO
boundaries = [
    ([0, 120, 0], [140, 255, 100]), #VERDE Izquierda
    ([25, 0, 75], [180, 38, 255]) #ROJO Derecha
]


def identifyHand(img,landmark,width,height,label):
    maxHeight = int((max(landmark, key = lambda p: p.y).y)*height)+20
    minHeight = int((min(landmark, key = lambda p:p.y).y)*height)-20
    left =int((min(landmark, key = lambda p:p.x).x) *width)-20
    right = int((max(landmark, key = lambda p:p.x).x)*width)+20
    onlyHands = {
        'img':img[minHeight:maxHeight,left:right],
        'type': label
    }
    return onlyHands

def getHands(img,width,height,results):
    handImgs = []
    if results.multi_hand_landmarks is not None:
        for i,hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks( frame,hand_landmarks,mp_hands.HAND_CONNECTIONS)
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
    
def handsegment(frame):
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

def changeColorHands(img):
    skinColor = np.array([72,79,85], dtype='uint8')
    green = np.array([59,169,54],  dtype='uint8')
    red =  np.array([133,36,59],  dtype='uint8')
    cv2.colorChange(green,skinColor,img)
    cv2.colorChange(red,skinColor,img)
    return img
with mp_hands.Hands(
    static_image_mode= False,
    max_num_hands= 2,
    min_detection_confidence= 0.8,
    min_tracking_confidence = 0.8

) as hands:
    while cap.isOpened():
        ret,frame = cap.read()
        if ret == False:
            break
        frame = cv2.resize(frame,(0,0),frame,0.4,0.4)
        height,width,_ = frame.shape
        frame = cv2.flip(frame,1)
        #frame = changeColorHands(frame)
        results = hands.process(frame)
        hands_img = getHands(frame,width,height,results)        
        if len(hands_img) > 0:
            for hand in hands_img:
                if (0 not in hand['img'].shape):
                    cv2.imshow(f"{hand['type']} hand",hand['img'])
        cleanHandWindows(hands_img)
        frame = handsegment(frame)
        cv2.imshow("Image", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
cap.release()
cv2.destroyAllWindows()