import torch
import cv2
import glob
import pandas as pd
import numpy as np
model = torch.hub.load('./yolov5-master','custom', path='./pt/best.pt',force_reload=True,source="local")

img = './images/*' 
ext = ['png', 'jpg','jpeg']
files = []
imdir = './images/'
[files.extend(glob.glob(imdir + '*.' + e)) for e in ext]
images = [cv2.imread(file) for file in files]

a=1
for i in images:
    print("Image"+str(a))
    results = model(i)
    toPandas=results.pandas()
    predictions=results.pandas().xyxy[0]
    print("Length of predictions =>",len(predictions))
    xmin=predictions["xmin"]
    ymin=predictions["ymin"]
    xmax=predictions["xmax"]
    ymax=predictions["ymax"]
    accuracy=predictions["confidence"]
    if len(predictions)>0:
        for j in range(len(predictions)):
            print(str(j),".th prediction;")
            print("xmin",xmin[j])
            print("ymin",ymin[j])
            print("xmax",xmax[j])
            print("ymax",ymax[j])
            print("accuracy", accuracy[j])
            i=np.array(i)
            a=[int(xmin[j]),int(ymin[j])]
            b=[int(xmax[j]),int(ymin[j])]
            c=(int(xmax[j]),int(ymax[j]))
            d=(int(xmin[j]),int(ymax[j]))
            newAccuracy=str(accuracy[j])[0:4]
            cv2.rectangle(i,a,c,(0,255,0),2)
            cv2.putText(i,"Accuracy: "+newAccuracy,a,cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),1,cv2.LINE_AA)
    cv2.putText(i,"{} object detected!".format(len(predictions)),(20,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1,cv2.LINE_AA)
    cv2.imshow(str(a),i)
    print("\n\n")

cv2.waitKey(0)
cv2.destroyAllWindows()
