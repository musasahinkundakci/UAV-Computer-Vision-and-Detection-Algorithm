import torch
import cv2
import glob
model = torch.hub.load('ultralytics/yolov5','custom', path='./pt/best.pt',force_reload=True)

img = './images/*'  # or file, Path, PIL, OpenCV, numpy, list
ext = ['png', 'jpg','jpeg']
files = []
imdir = './images/'
[files.extend(glob.glob(imdir + '*.' + e)) for e in ext]
images = [cv2.imread(file) for file in files]
print(images)
a=2
for i in images:
    print("aaa")
    results = model(i)
    results.save() 
    cv2.imshow(str(a),results.imgs[0])

    a+=1
cv2.waitKey(0)
cv2.destroyAllWindows()