import cv2
from deepface import DeepFace

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    #demography = DeepFace.analyze("juan.jpg", actions = ['age', 'gender', 'race', 'emotion'])

    result = DeepFace.analyze(img_path=frame, actions=['emotion','age','gender','race'], enforce_detection=False)
    
    #result = DeepFace.analyze(img_path=frame, actions=['emotion'], enforce_detection=False)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)

        #emotionz = result["dominant_emotion"]
        #txt = str(result['dominant_emotion'])
        txt = str(result[0]['dominant_emotion'])
        txt2 = str(result[0]['age'])
        txt3 = str(result[0]['gender'])

        cv2.putText(frame, txt, (x, y+3), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, txt2, (x, y+6), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, txt3, (x, y+9), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('frame', frame)
    print(result)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
