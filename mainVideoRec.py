import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture('./prueba.mp4')


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
        height,width,_ = frame.shape
        frame = cv2.resize(frame,(0,0),frame,0.4,0.4)
        frame = cv2.flip(frame,1)

        frame_rgb = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

        results = hands.process(frame_rgb)
        print("HANDEDNESS:",results.multi_handedness)

        if results.multi_hand_landmarks is not None:

            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks( frame,hand_landmarks,mp_hands.HAND_CONNECTIONS)
                #mp_drawing.DrawingSpec(color = (0,255,255), thickness = 4, circle_radius = 5),
                #mp_drawing.DrawingSpec(color = (255,0,0), thickness = 4))            

        cv2.imshow("Frame",frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()