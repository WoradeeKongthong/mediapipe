"""
Pose Detection
"""
import mediapipe as mp
import cv2

# create drawer
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# create pose detector
mp_pose = mp.solutions.pose

# load image
img = cv2.imread('data/person.jpg')
img = cv2.resize(img, (int(img.shape[1]/2.5), int(img.shape[0]/2.5)))

with mp_pose.Pose() as pose:
    
    results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    if results.pose_landmarks:
        print("detected")
        
        # get landmarks (normalized with image shape)
        landmark_list = []
        for landmark in results.pose_landmarks.landmark :
            x, y, z, visibility = landmark.x, landmark.y, landmark.z, landmark.visibility
            landmark_list.append([x,y,z,visibility])
            
        # visualize landmarks
        mp_drawing.draw_landmarks(img, 
                                  results.pose_landmarks, 
                                  mp_pose.POSE_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()) # drawing_spec
        
cv2.imshow("result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# list elements
landmark_name_list = list(mp_pose.PoseLandmark)
