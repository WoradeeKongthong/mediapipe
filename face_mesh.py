"""
Face Mesh with MediaPipe
"""
import mediapipe as mp
import cv2

# create face mesh detector
mp_face_mesh = mp.solutions.face_mesh

# create drawer
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=2, color=(0,255,0))

# load image
img = cv2.imread('data/face.jpg')

# create mesh
with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.5) as face_mesh:
    
    results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    if results.multi_face_landmarks :
        for face_landmarks in results.multi_face_landmarks:
            
            # extract 468 landmarks
            ih, iw, ic = img.shape
            landmarks = []
            for i,lm in enumerate(face_landmarks.landmark):
                x, y = int(lm.x * iw), int(lm.y * ih)
                landmarks.append([x,y])
            
            # visualize mesh
            mp_drawing.draw_landmarks(
              image=img,
              landmark_list=face_landmarks,
              connections=mp_face_mesh.FACEMESH_TESSELATION, #FACEMESH_CONTOURS, FACEMESH_TESSELATION
              landmark_drawing_spec=drawing_spec, #None, drawing_spec
              connection_drawing_spec=mp_drawing_styles
              .get_default_face_mesh_tesselation_style())

# draw extracted landmarks
# for id,lm in enumerate(landmarks) :
#     cv2.circle(img, (lm[0], lm[1]), 2, (0,255,0), -1)
#     cv2.putText(img, str(id), (lm[0],lm[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0,0,255),1)
    
cv2.imshow("result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()