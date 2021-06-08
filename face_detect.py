import cv2
import time
import mediapipe as mp

cap = cv2.VideoCapture(0) 

mp_drawing = mp.solutions.drawing_utils
mp_face_detection = mp.solutions.face_detection

face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)



while True: 
    ret, frame = cap.read() 
    
    results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.detections:
        for detection in results.detections:
            mp_drawing.draw_detection(frame, detection)

            x1 = detection.location_data.relative_bounding_box.xmin # left side of face bounding box
            x2 = x1 + detection.location_data.relative_bounding_box.width # right side of face bounding box

            cx = (x1 + x2) / 2
    
    cv2.imshow('test', frame) 
    cv2.waitKey(1) 

    
cap.release() 
# destroyAllWindows : 화면에 나타난 윈도우를 종료
cv2.destroyAllWindows()


#http://blog.naver.com/PostView.nhn?blogId=skyjjw79&logNo=222327014865&parentCategoryNo=54&categoryNo=60&viewDate=&isShowPopularPosts=false&from=postView
