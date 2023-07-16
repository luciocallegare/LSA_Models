import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

with mp_hands.Hands(
    static_image_mode= True,
    max_num_hands= 2,
    min_detection_confidence= 0.5,
    min_tracking_confidence = 0.5
) as hands:
    
    image = cv2.imread('./mano2.jpg')
    height,width,_ = image.shape
    image = cv2.flip(image,1)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = hands.process(image_rgb)

    print(results)

    #HANDEDNESS
    #print("Handedness: ", results.multi_handedness)

    #HAND LANDMARKS
    #print(':::::::::LANDMARKS MANO 1:::::::::::::::')
    #print(type(results.multi_hand_landmarks[0]))

    #print(':::::::LANDMARKS MANO 2:::::::::::::::::')
    #print(results.multi_hand_landmarks[1])
    if results.multi_hand_landmarks is not None:
        #---------------------------------------------------#
        #Dibujando puntos y sus conexiones con mediapipe
        print("LANDMARKS:::::::")
        indexes = [4,8,12,16,20]
        print(indexes)
        for hand_landmarks in results.multi_hand_landmarks:  ### ITERO MANOS
            mp_drawing.draw_landmarks( image,hand_landmarks,mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color = (0,255,255), thickness = 4, circle_radius = 5),
            mp_drawing.DrawingSpec(color = (255,0,0), thickness = 4))

            #Accedo a las punta del pulgar 

            #x1 = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x*width)
            #y1 = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y*height)

            #Accedo por indices
            for (i,points) in enumerate(hand_landmarks.landmark):
                if i in indexes:
                    x = int(points.x * width)
                    y = int(points.y * height)
                    image = cv2.circle(image,(x,y),3,(0,0,255),3)
                    print(x,y)


    image = cv2.flip(image,1)

cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

