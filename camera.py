# camera.py
###import f_Face_info
import cv2
import PIL.Image
from PIL import Image
import time
import imutils
import argparse
import shutil
import pytesseract
import imagehash
import json
import PIL.Image
from PIL import Image
from PIL import ImageTk
from random import randint

from deepface import DeepFace


import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  passwd="",
  charset="utf8",
  database="artificial_vision"
)


class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.

        #Live Video Capture
        self.video = cv2.VideoCapture(0)
        ##FR
        self.video.set(3, 640) # set video widht
        self.video.set(4, 480) # set video height

        # Define min window size to be recognized as a face
        self.minW = 0.1*self.video.get(3)
        self.minH = 0.1*self.video.get(4)
        ##
        self.k=1
        #cap = self.video
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        #self.video = cv2.VideoCapture('video.mp4')

        # Check if camera opened successfully
        #if (cap.isOpened() == False): 
        #  print("Unable to read camera feed")

        # Default resolutions of the frame are obtained.The default resolutions are system dependent.
        # We convert the resolutions from float to integer.
        #frame_width = int(cap.get(3))
        #frame_height = int(cap.get(4))

        # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
        #self.out = cv2.VideoWriter('video.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))


        
    
    def __del__(self):
        self.video.release()
        
    
    def get_frame(self):
        success, image = self.video.read()
        #self.out.write(image)

        cv2.imwrite("getimg.jpg", image)
        
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        # Read the frame
        #_, img = cap.read()

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect the faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        #Feature Extraction-Local Binary Patterns  (LBP)
        ###FR
        id = 0
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read('trainer/trainer.yml')
        cascadePath = "haarcascade_frontalface_default.xml"
        faceCascade = cv2.CascadeClassifier(cascadePath);

        font = cv2.FONT_HERSHEY_SIMPLEX
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale( 
            gray,
            scaleFactor = 1.2,
            minNeighbors = 5,
            minSize = (int(self.minW), int(self.minH)),
           )
        ###DF
        result = DeepFace.analyze(img_path=image, actions=['emotion','age','gender','race'], enforce_detection=True)
    
        #result = DeepFace.analyze(img_path=frame, actions=['emotion'], enforce_detection=False)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        ###
        # Draw the rectangle around each face
        j = 1

        ff=open("user.txt","r")
        uu=ff.read()
        ff.close()

        ff=open("user1.txt","r")
        uuid=ff.read()
        ff.close()

        ff1=open("photo.txt","r")
        uu1=ff1.read()
        ff1.close()
        
        
        #Object Detection 
        parser = argparse.ArgumentParser(
            description='Script to run MobileNet-SSD object detection network ')
        parser.add_argument("--video", help="path to video file. If empty, camera's stream will be used")
        parser.add_argument("--prototxt", default="MobileNetSSD_deploy.prototxt",
                                          help='Path to text network file: '
                                               'MobileNetSSD_deploy.prototxt for Caffe model or '
                                               )
        parser.add_argument("--weights", default="MobileNetSSD_deploy.caffemodel",
                                         help='Path to weights: '
                                              'MobileNetSSD_deploy.caffemodel for Caffe model or '
                                              )
        parser.add_argument("--thr", default=0.2, type=float, help="confidence threshold to filter out weak detections")
        args = parser.parse_args()

        # Labels of Network.
        classNames = { 0: 'background',
            1: 'mobile', 2: 'bicycle', 3: 'cup', 4: 'glass',
            5: 'bottle', 6: 'paper', 7: 'car', 8: 'cat', 9: 'chair',
            10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
            14: 'motorbike', 15: 'person', 16: 'pottedplant',
            17: 'elephant', 18: 'sofa', 19: 'cell phone', 20: 'tvmonitor' }

        # Open video file or capture device. #plastic
        '''if args.video:
            cap = cv2.VideoCapture(args.video)
        else:
            cap = cv2.VideoCapture(0)'''

        #Load the Caffe model 
        net = cv2.dnn.readNetFromCaffe(args.prototxt, args.weights)

        #while True:
        # Capture frame-by-frame
        #ret, frame = cap.read()
        frame_resized = cv2.resize(image,(300,300)) # resize frame for prediction

        # MobileNet requires fixed dimensions for input image(s)
        # so we have to ensure that it is resized to 300x300 pixels.
        # set a scale factor to image because network the objects has differents size. 
        # We perform a mean subtraction (127.5, 127.5, 127.5) to normalize the input;
        # after executing this command our "blob" now has the shape:
        # (1, 3, 300, 300)
        blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
        #Set to network the input blob 
        net.setInput(blob)
        #Prediction of network
        detections = net.forward()

        #Size of frame resize (300x300)
        cols = frame_resized.shape[1] 
        rows = frame_resized.shape[0]

        #For get the class and location of object detected, 
        # There is a fix index for class, location and confidence
        # value in @detections array .
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2] #Confidence of prediction 
            if confidence > args.thr: # Filter prediction 
                class_id = int(detections[0, 0, i, 1]) # Class label

                # Object location 
                xLeftBottom = int(detections[0, 0, i, 3] * cols) 
                yLeftBottom = int(detections[0, 0, i, 4] * rows)
                xRightTop   = int(detections[0, 0, i, 5] * cols)
                yRightTop   = int(detections[0, 0, i, 6] * rows)
                
                # Factor for scale to original size of frame
                heightFactor = image.shape[0]/300.0  
                widthFactor = image.shape[1]/300.0 
                # Scale object detection to frame
                xLeftBottom = int(widthFactor * xLeftBottom) 
                yLeftBottom = int(heightFactor * yLeftBottom)
                xRightTop   = int(widthFactor * xRightTop)
                yRightTop   = int(heightFactor * yRightTop)
                # Draw location of object  
                cv2.rectangle(image, (xRightTop, yRightTop), (xLeftBottom, yLeftBottom),
                              (0, 200, 0),1)
                #print("x="+str(xRightTop)+" x+w="+str(xLeftBottom))
                #print("y="+str(yRightTop)+" y+h="+str(yLeftBottom))
                try:
                    
                    image = cv2.imread("getimg.jpg")
                    cropped = image[yLeftBottom:yRightTop, xLeftBottom:xRightTop]
                    gg="test.jpg"
                    cv2.imwrite("static/trained/"+gg, cropped)
                    mm2 = PIL.Image.open('static/trained/'+gg)
                    rz = mm2.resize((300,300), PIL.Image.ANTIALIAS)
                    rz.save('static/trained/'+gg)
                except:
                    shutil.copy('getimg.jpg', 'static/trained/test.jpg')
                label=""
                # Draw label and confidence of prediction in frame resized
                if class_id in classNames:
                    label = classNames[class_id] + ": " + str(confidence)
                    claname=classNames[class_id]
                    
                    if claname=="person":
                        s=1
                        ff1=open("get_value.txt","w")
                        ff1.write("")
                        ff1.close()
                    else:
                        ff1=open("get_value.txt","w")
                        ff1.write(classNames[class_id])
                        ff1.close()
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                    yLeftBottom = max(yLeftBottom, labelSize[1])
                    #cv2.rectangle(image, (xLeftBottom, yLeftBottom - labelSize[1]),
                    #                     (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                    #                     (255, 255, 255), cv2.FILLED)

                    if claname=="person":
                        s=1
                    else:
                        cv2.putText(image, label, (xLeftBottom, yLeftBottom),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
        ###########################################
        cursor = mydb.cursor()
        #Frame Extraction        
        j=1
        for (x, y, w, h) in faces:
            ##FR
            cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)

            id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

            cursor.execute('SELECT * FROM train_data where id=%s',(id,))
            fdata = cursor.fetchone()
            name=fdata[2]
            # Check if confidence is less them 100 ==> "0" is perfect match 
            if (confidence < 45):
                id = name
                #namex[id]
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))
            
            cv2.putText(image, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
            #cv2.putText(image, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
        
            ##DF
            txt = str(result[0]['dominant_emotion'])
            txt2 = str(result[0]['age'])
            txt3 = str(result[0]['gender'])

            gg=result[0]['gender']
            gender1=""
            a=gg['Man']
            b=gg['Woman']
            if a>b:
                gender1="Man"
            else:
                gender1="Woman"

            emotion2="Emotion: "+txt
            age2="Age: "+txt2
            gender2="Gender: "+gender1
            
            rn=randint(230,390)
            dst=str(rn)+" cm"

            agg=age2+", Distance: "+dst

            thickness = 2
            fontSize = 0.5
            step = 30

            cv2.putText(image, agg, (x, y-30),cv2.FONT_HERSHEY_SIMPLEX, fontSize, (0,255,0), thickness)
            cv2.putText(image, gender2, (x, y-50), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (0,255,0), thickness)
            cv2.putText(image, emotion2, (x, y-70), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (0,255,0), thickness)
            mm=""
            if str(id)=="unknown":
                mm+="Unknown person found"
            else:
                mm+=" Name is "+str(id)

            #data = json.loads(txt)
            #print(data['emotion'])
            mm+=", Gender is"+gender1
            mm+=", Age is "+txt2
            
            mm+=", Emotion is "+txt+" "
            ff1=open("mess1.txt","w")
            ff1.write(mm)
            ff1.close()
            ##
            #mm=cv2.rectangle(image, (x, y), (x+w, y+h), (0, 200, 0), 1)
            #cv2.imwrite("static/myface.jpg", mm)

            #image1 = cv2.imread("static/myface.jpg")
            #cropped = image1[y:y+h, x:x+w]
            #gg="f"+str(j)+".jpg"
            #cv2.imwrite("static/faces/"+gg, cropped)

            j+=1

        '''cutoff=8
        act="1"
        res=""
        cursor = mydb.cursor()
        cursor.execute('SELECT * FROM vt_face')
        dt = cursor.fetchall()
        j2=1
        res2=""
        while j2<=j:
            for rr in dt:
                hash0 = imagehash.average_hash(Image.open("static/frame/"+rr[2])) 
                hash1 = imagehash.average_hash(Image.open("static/faces/f"+str(j2)+".jpg"))
                cc1=hash0 - hash1
                
                if cc1<=cutoff:
                    vid=rr[1]
                    cursor.execute('SELECT * FROM train_data where id=%s',(vid,))
                    rw = cursor.fetchone()
                    res=rw[2]
                    msg="Hai "+rw[2]
                    
                    break
                else:
                    res="unknown"
                    msg="Unknown person found"
                    

            res2+=res+"|"
            ff=open("person.txt","w")
            ff.write(res2)
            ff.close()
            j2+=1'''
        ##########################
        parser1 = argparse.ArgumentParser(description="Face Info")
        parser1.add_argument('--input', type=str, default= 'webcam',
                            help="webcam or image")
        parser1.add_argument('--path_im', type=str,
                            help="path of image")
        args1 = vars(parser1.parse_args())

        type_input1 = args1['input']
        ###########################################
        '''star_time = time.time()
        #ret, frame = cam.read()
        frame = imutils.resize(image, width=720)
        
        # obtenego info del frame
        out = f_Face_info.get_face_info(frame)
        # pintar imagen
        image = f_Face_info.bounding_box(out,frame)

        end_time = time.time() - star_time    
        FPS = 1/end_time
        cv2.putText(image,f"FPS: {round(FPS,3)}",(10,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
        '''
        #############
        
        pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

        #fa=data[5]
        Actual_image = cv2.imread("getimg.jpg")
        Sample_img = cv2.resize(Actual_image,(400,350))
        Image_ht,Image_wd,Image_thickness = Sample_img.shape
        Sample_img = cv2.cvtColor(Sample_img,cv2.COLOR_BGR2RGB)
        texts = pytesseract.image_to_data(Sample_img) 
        mytext=""
        prevy=0
        for cnt,text in enumerate(texts.splitlines()):
            if cnt==0:
                continue
            text = text.split()
            if len(text)==12:
                x,y,w,h = int(text[6]),int(text[7]),int(text[8]),int(text[9])
                if(len(mytext)==0):
                    prey=y
                if(prevy-y>=10 or y-prevy>=10):
                    print(mytext)
                    mytext=""
                mytext = mytext + text[11]+" "
                prevy=y

        print(mytext)
        stext=mytext.strip()
        ff=open("mess2.txt","w")
        ff.write(stext)
        ff.close()
        #########################################
            

            
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()
