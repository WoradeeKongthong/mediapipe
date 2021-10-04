"""
Face Detection on video with MediaPipe
"""
import cv2
import mediapipe as mp

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture('data/people.mp4')

while cap.isOpened():
    check, img = cap.read()
    if check :
        
        # face detection
        with mpFaceDetection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as detector :
            results = detector.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            if results.detections :
                for detection in results.detections:
                    # extract bounding box from detection
                    bboxD = detection.location_data.relative_bounding_box
                    ih, iw, ic = img.shape
                    x, y, w, h = int(bboxD.xmin * iw), int(bboxD.ymin * ih),\
                        int(bboxD.width * iw), int(bboxD.height * ih)
            
                    # draw detections
                    # option 1
                    mpDraw.draw_detection(img, detection)
                    # option 2
                    # cv2.rectangle(img, (x,y), (x+w, y+h), (255,255,255), 1)
                
        cv2.imshow("result", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
