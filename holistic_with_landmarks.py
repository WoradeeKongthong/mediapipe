"""
Holistic : Pose + Face + Hands
"""
import mediapipe as mp
import cv2

# create drawer
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# create holistic object
mp_holistic = mp.solutions.holistic

# load image
img = cv2.imread('data/person1.jpg')
img = cv2.resize(img, (int(img.shape[1]/2.5), int(img.shape[0]/2.5)))

with mp_holistic.Holistic() as holistic :
    
    results = holistic.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    if results.pose_landmarks:
        # extract pose landmarks
        pose_landmark_list = []
        for pose_landmark in results.pose_landmarks.landmark:
            x,y,z,visibility = pose_landmark.x, pose_landmark.y, pose_landmark.z, pose_landmark.visibility
            pose_landmark_list.append([x,y,z,visibility])
    
    if results.face_landmarks:
        # extract face landmarks
        face_landmark_list = []
        for face_landmark in results.face_landmarks.landmark:
            x,y,z = face_landmark.x, face_landmark.y, face_landmark.z
            face_landmark_list.append([x,y,z])
        
    if results.left_hand_landmarks:
        # extract left hand landmarks
        left_hand_landmark_list = []
        for left_hand_landmark in results.left_hand_landmarks.landmark:
            x,y,z = left_hand_landmark.x, left_hand_landmark.y, left_hand_landmark.z
            left_hand_landmark_list.append([x,y,z])
    
    if results.right_hand_landmarks:
        # extract right hand landmarks
        right_hand_landmark_list = []
        for right_hand_landmark in results.right_hand_landmarks.landmark:
            x,y,z = right_hand_landmark.x, right_hand_landmark.y, right_hand_landmark.z
            right_hand_landmark_list.append([x,y,z])
        
        # draw face mesh
        mp_drawing.draw_landmarks(img,
                                  results.face_landmarks,
                                  mp_holistic.FACEMESH_TESSELATION,
                                  landmark_drawing_spec=None,
                                  connection_drawing_spec=mp_drawing_styles
                                  .get_default_face_mesh_tesselation_style())
        
        # draw pose landmarks
        mp_drawing.draw_landmarks(img, 
                                  results.pose_landmarks,
                                  mp_holistic.POSE_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing_styles
                                  .get_default_pose_landmarks_style())
        
        # draw left hand landmarks
        mp_drawing.draw_landmarks(img,
                                  results.left_hand_landmarks,
                                  mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing_styles.get_default_hand_landmarks_style(),
                                  mp_drawing_styles.get_default_hand_connections_style())
        
        # draw right hand landmarks
        mp_drawing.draw_landmarks(img,
                                  results.right_hand_landmarks,
                                  mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing_styles.get_default_hand_landmarks_style(),
                                  mp_drawing_styles.get_default_hand_connections_style())
        

    
cv2.imshow("result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()