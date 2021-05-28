import numpy as np
import cv2
from keras.models import load_model
from keras.preprocessing import image
from numpy.core.fromnumeric import argsort

class_name=["Paper","Rock","scissor"]
model=load_model("RPS-Game\RPS-model.h5")

cam=cv2.VideoCapture(0)
while True:
    con,img=cam.read()
    imga=img
    cv2.rectangle(imga,(0,0),(300,300),(0,0,255),2)
    cv2.putText(imga,"Put your hand in the box",(0,400), cv2.HOUGH_GRADIENT_ALT, 0.8,(0,0,0), 2)
    img=img[0:300,0:300]
    img=cv2.resize(img,(150,150))
    img=image.img_to_array(img)
    img=img/255
    img=np.expand_dims(img,axis=0)
    if cv2.waitKey(1)==13:
        break
    else:
        l=model.predict(img,batch_size=10)
        ans=class_name[np.argsort(l[0])[2]]
        cv2.putText(imga,ans,(350,150),cv2.HOUGH_GRADIENT,0.8,(0,0,0),2)
        cv2.imshow("test",imga)


#for testing using a saved image 
'''
img=image.load_img("RPS-Game/rps/scissors/scissors01-002.png",target_size=(150,150))
img=image.img_to_array(img)
img=img/255
img=np.expand_dims(img,axis=0) 
l=model.predict(img)
print(class_name[np.argmax(l[0])])
'''