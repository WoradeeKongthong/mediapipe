"""
Face Detection with MediaPipe
"""
import mediapipe as mp
import cv2

# create face detector
mpFaceDetection = mp.solutions.face_detection

# create drawer
mpDrawer = mp.solutions.drawing_utils

# load image
img = cv2.imread('data/people.jpg')
img = cv2.resize(img, (int(img.shape[1]/3), int(img.shape[0]/3)))

with mpFaceDetection.FaceDetection(
        model_selection=1, 
        min_detection_confidence=0.5) as detector :
    results = detector.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if results.detections :

        annotated_img = img.copy()
        
        for id, detection in enumerate(results.detections) :
            
            # extract bounding box from detection
            bboxD = detection.location_data.relative_bounding_box
            ih, iw, ic = annotated_img.shape
            x, y, w, h = int(bboxD.xmin * iw), int(bboxD.ymin * ih),\
                int(bboxD.width * iw), int(bboxD.height * ih)
            
            # extract landmarks (ight eye, left eye, nose tip, mouth center, 
            # right ear tragion, and left ear tragion)
            lm = detection.location_data.relative_keypoints
            
            # draw detection on image
            # option 1 
            mpDrawer.draw_detection(annotated_img, detection)
            # option 2
            # cv2.rectangle(annotated_img, (x,y), (x+w, y+h), (255,255,255), 1)

cv2.imshow("press q to exit", annotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
        