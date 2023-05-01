import cv2 as cv
import numpy as np

# Device Camera Capture
# cap = cv.VideoCapture(0)

# Webcam Capture
# cap = cv.VideoCapture(1)

cap = cv.VideoCapture("video.mp4")

whT = 320
confThreshold =0.5
nmsThreshold= 0.2

classesFile = "names.txt"
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().split('\n')

windowName = "Object Detection"

# FPS - 45
# Download it from - https://pjreddie.com/darknet/yolo/
# modelConfiguration = "yolov3-320.cfg"
# modelWeights = "yolov3-320.weights"

# FPS - 220
modelConfiguration = "yolov3-tiny.cfg"
modelWeights = "yolov3-tiny.weights"

net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

def findObjects(outputs,img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w,h = int(det[2]*wT) , int(det[3]*hT)
                x,y = int((det[0]*wT)-w/2) , int((det[1]*hT)-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))

    indices = cv.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

    for i in indices:
        i = i
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv.rectangle(img, (x, y), (x+w,y+h), (255, 102, 94), 2)
        cv.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',
                  (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 102, 94), 2)

while True:
    success, img = cap.read()
    blob = cv.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    layersNames = net.getLayerNames()
    outputNames = [(layersNames[i - 1]) for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)
    findObjects(outputs,img)
    cv.imshow(windowName, img)
    cv.waitKey(1)
    if cv.getWindowProperty(windowName, cv.WND_PROP_VISIBLE) <1:
        break

cv.destroyAllWindows()