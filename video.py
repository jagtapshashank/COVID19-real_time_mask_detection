import cv2
import pickle
import numpy as np

# Load the trained model from the pickle file
with open('mask_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Starting webcam
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot Open Webcam")

while True:
    ret,frame = cap.read()
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,1.1,4)

    for x,y,w,h in faces:
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = frame[y:y+h,x:x+w]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        facess = faceCascade.detectMultiScale(roi_gray)
        if len(facess) == 0:
            print("Face not detected")
        else:
            for (ex,ey,ew,eh) in facess:
                face_roi = roi_color[ey:ey+eh,ex:ex+ew]
        final_image = cv2.resize(frame,(224,224))
        final_image = np.expand_dims(final_image,axis = 0)   # fourth dimension
        final_image = final_image/255.0
        font = cv2.FONT_HERSHEY_SIMPLEX
        Predictions = model.predict(final_image)
        predicted_label = np.argmax(Predictions, axis=1)[0]
        
        font_scale = 1.5
        font = cv2.FONT_HERSHEY_PLAIN

        if(predicted_label == 1):     # Without Mask
            status = "NO FACE MASK :( "
            x1,y1,w1,h1 = 0,0,175,175
            cv2.putText(frame,status,(100,150),font,3,(0,0,255),2,cv2.LINE_4)
        else:                       # With Mask
            status = "FACE MASK ON :) "
            x1,y1,w1,h1 = 0,0,175,175
            cv2.putText(frame,status,(100,150),font,3,(0,255,0),2,cv2.LINE_4)
            
        cv2.imshow("FACE MASK DETECTION",frame)
        
        if cv2.waitKey(2) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
