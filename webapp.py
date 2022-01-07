from logging import debug
from flask import Flask, render_template
from flask.wrappers import Response
import cv2
app = Flask(__name__)

cap = cv2.VideoCapture(0) 

import cv2
import numpy as np
from keras.models import load_model
model=load_model("./model2-010.model")

results={0:'without mask',1:'mask'}
GR_dict={0:(0,0,255),1:(0,255,0)}

rect_size = 4
cap = cv2.VideoCapture(0) 


haarcascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def gen_frames():
    while True:
        rval, im = cap.read()
        im=cv2.flip(im,1,1) 
        if not rval:
            break
        else:
        
            rerect_size = cv2.resize(im, (im.shape[1] // rect_size, im.shape[0] // rect_size))
            faces = haarcascade.detectMultiScale(rerect_size)
            for f in faces:
                (x, y, w, h) = [v * rect_size for v in f] 
                
                face_img = im[y:y+h, x:x+w]
                rerect_sized=cv2.resize(face_img,(150,150))
                normalized=rerect_sized/255.0
                reshaped=np.reshape(normalized,(1,150,150,3))
                reshaped = np.vstack([reshaped])
                result=model.predict(reshaped)

                
                label=np.argmax(result,axis=1)[0]
            
                cv2.rectangle(im,(x,y),(x+w,y+h),GR_dict[label],2)
                cv2.rectangle(im,(x,y-40),(x+w,y),GR_dict[label],-1)
                cv2.putText(im, results[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
            
            ret, buffer = cv2.imencode('.jpg', im)
            frame = buffer.tobytes()
            yield(b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            # cv2.imshow('LIVE',   im)
            # key = cv2.waitKey(10)
            
            # if key == 27: 
            #     break

# cap.release()

# cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)

