import cv2
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
camera = cv2.VideoCapture(0)
while True: 
    ret,frame = camera.read()
    greyscale = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(greyscale,1.1,5)
    eyes = eye_cascade.detectMultiScale(greyscale,1.1,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    for (x,y,w,h) in eyes:
        cv2.rectangle(frame, (x,y),(x+w, y+h), (0,40,255),2)
    cv2.imshow('faceDetection',frame)

    if cv2.waitKey(1) == 32: break

camera.release()
cv2.destroyAllWindows()