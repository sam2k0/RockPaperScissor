import os
import cv2
import numpy as np
def CreatDataset():    
    # run this code once for folder creation
    """" 
    folder_name="custom_dataset"
    os.mkdir("RPS-Game/"+folder_name)
    rock="RPS-Game/"+folder_name+"/rock"
    paper="RPS-Game/"+folder_name+"/paper"
    scissor="RPS-Game/"+folder_name+"/scissor"
    os.mkdir(rock)
    os.mkdir(paper)
    os.mkdir(scissor)
    """

    cam=cv2.VideoCapture(0)
    count=0
    flag=0
    while True:
        con,img=cam.read()
        if con :
            cv2.putText(img,"Hit ENTER to capture images for training dataset",(0,440),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),2)
            cv2.rectangle(img,(0,0),(300,300),(0,0,255),2)
            
            cv2.putText(img,"Note: Make sure that box contains plain background ",(0,460),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)
            cv2.imshow("Creating dataset...",img)
            imag=img[0:300,0:300]
            
            if cv2.waitKey(30)==13:
                flag=flag+1
                count=0
            if flag==1 and count<500:
                cv2.putText(img,"Rock images saving..... {}".format(count),(0,350),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2)
                cv2.imshow("Creating dataset...",img)
                cv2.imwrite("RPS-Game/custom_dataset/rock/rock{}.jpg".format(count),imag)
            if flag==2 and count<500:
                cv2.putText(img,"Paper images saving.....{}".format(count),(0,350),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2)
                cv2.imshow("Creating dataset...",img)
                cv2.imwrite("RPS-Game/custom_dataset/paper/paper{}.jpg".format(count),imag)
            if flag==3 and count<500:
                cv2.putText(img,"Scissor images saving.....{}".format(count),(0,350),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2)
                cv2.imshow("Creating dataset...",img)
                cv2.imwrite("RPS-Game/custom_dataset/scissor/scissor{}.jpg".format(count),imag)
            if flag==4:
                break
            
            count=count+1

        else:
            print("camera not detected")            


CreatDataset()