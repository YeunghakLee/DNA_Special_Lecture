import cv2



#괄호안에 파일명을 쓰면 파일이 로드됌

#detecting한 얼굴을 표시할 폰트 정의

#cap = cv2.VideoCapture('newface.mp4') #카메라 생성

font = cv2.FONT_HERSHEY_SIMPLEX



#create the window & change the window size

#윈도우 생성 및 사이즈 변경

cv2.namedWindow('Face')



#haar 코드 사용(frontal_face) -> 어떤 파일을 쓰느냐에 따라 인식할 객체가 달라짐

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
frame = cv2.imread('BTS.png', cv2.IMREAD_COLOR)
grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#faces = face_cascade.detectMultiScale(grayframe, 1.8, 2, 0, (30, 30))
faces = face_cascade.detectMultiScale(grayframe, 1.3, 5)
#eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


for (x,y,w,h) in faces:

        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0), 2)

        #cv2.putText(frame, 'Detected Face', (x-5, y-5), font, 0.9, (255,255,0),2)
        #roi_color = frame[y:y+h, x:x+w]
        #eyes = eye_cascade.detectMultiScale(roi_gray)
        #for (ex,ey,ew,eh) in eyes:
        #    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)


cv2.imshow('Face',frame)

cv2.waitKey(0)
cv2.destroyAllWindows()

