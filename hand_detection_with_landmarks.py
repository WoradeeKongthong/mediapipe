"""
Hand Detection and Landmarks with MediaPipe
"""
import mediapipe as mp
import cv2

# create hand detector
mp_hands = mp.solutions.hands

# list index and landmarks
landmark_name_list = list(mp_hands.HandLandmark)

# create drawer
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# load image
img = cv2.imread('data/Hand-2.jpg')
#img = cv2.flip(img, 1)

with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5) as hands:
    
    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            
            # extract information
            landmark_list = []
            for lm in hand_landmarks.landmark:
                h, w, c = img.shape
                x, y = int(lm.x * w), int(lm.y * h)
                landmark_list.append([x,y])
            
            # drawing landmarks
            mp_drawing.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

cv2.imshow("result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# explore landmark (relative position with image size)
print('hand_landmarks:', hand_landmarks)
print(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP])
            