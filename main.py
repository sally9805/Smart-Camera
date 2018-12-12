import cv2
import time
from PIL import Image
import numpy as np
import math


_COLOR_RED   = (0, 0, 255)
_COLOR_GREEN = (0, 255, 0)

def _draw_contours(array, *args, **kwargs):
    cv2.drawContours(array, *args, **kwargs)

def round_int(value):
        result = int(np.rint(value))
        return result

def _get_eucledian_distance(a, b):
    distance = math.sqrt( (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    return distance

def _get_defects_count(array, contour, defects):
    ndefects = 0
    for i in range(defects.shape[0]):
        s,e,f,_ = defects[i,0]
        beg     = tuple(contour[s][0])
        end     = tuple(contour[e][0])
        far     = tuple(contour[f][0])
        a       = _get_eucledian_distance(beg, end)
        b       = _get_eucledian_distance(beg, far)
        c       = _get_eucledian_distance(end, far)
        angle   = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) # * 57
            
        if angle <= math.pi/2 :#90:
            ndefects = ndefects + 1

            cv2.circle(array, far, 3, _COLOR_RED, -1)

        cv2.line(array, beg, end, _COLOR_RED, 1)

    return array, ndefects
    
def mount_roi(array, roi, color=(0, 255, 0), thickness=1):
        x, y, w, h = roi
        cv2.rectangle(array, (x, y), (x + w, y + h),
                      color=color, thickness=thickness)
        return array

def _get_contours(array):
    _, contours, _ = cv2.findContours(array, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    return contours

def _bodyskin_detetc(frame):
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    (_, cr, _) = cv2.split(ycrcb)
    cr1 = cv2.GaussianBlur(cr, (5, 5), 0)
    _, skin = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return skin

def _remove_background(frame):
    fgbg = cv2.createBackgroundSubtractorMOG2()
    fgmask = fgbg.apply(frame)

    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res

def grdetect(array):
    event      = None
    copy       = array.copy()
    array = _remove_background(array)
    thresh = _bodyskin_detetc(array)
    contours   = _get_contours(thresh.copy())
    largecont  = max(contours, key = lambda contour: cv2.contourArea(contour))
    roi_  = cv2.boundingRect(largecont)
    copy = mount_roi(copy, roi_, color = _COLOR_RED)
    convexHull = cv2.convexHull(largecont)
    _draw_contours(copy, contours    ,-1, _COLOR_RED  , 0)
    _draw_contours(copy, [largecont] , 0, _COLOR_GREEN, 0)
    _draw_contours(copy, [convexHull], 0, _COLOR_GREEN, 0)
    hull           = cv2.convexHull(largecont, returnPoints = False)
    defects        = cv2.convexityDefects(largecont, hull)
    if defects is not None:
        copy, ndefects = _get_defects_count(copy, largecont, defects)
        if   ndefects == 0:
            event = 0
        elif ndefects == 1:
            event = 2
        elif ndefects == 2:
            event = 3
        elif ndefects == 3:
            event = 4
        elif ndefects == 4:
            event = 5
    cv2.imshow('roi', copy)

    return event


def _crop_array(array, roi):

    x, y, w, h = roi
    crop       = array[ y : y + h , x : x + w ]

    return crop

def _get_roi(size, ratio):

    width, height = round_int(size[0] * ratio), round_int(size[1] * ratio)

    x = int(size[0]/2 - width/2)
    y = int(size[1]/2 - height/2)

    return (x, y, width, height)


def get_instruction(event_record):
    a = np.array(event_record)
    if np.max(a) == np.min(a):
        return np.max(a)
    else:
        return None

def combine(foreground, background, mask2):
    fore = foreground.copy()
    back = background.copy()
    h, w = back.shape[0:2]
    size = (300, 300)
    fore = cv2.resize(fore, size, interpolation=cv2.INTER_AREA)
    mask2 = cv2.resize(mask2, size, interpolation=cv2.INTER_AREA)
    back_ = back[h-size[1]:,w-size[0]:,:]
    #result = cv2.addWeighted(back_, 0.5, fore, 0.8, 0)
    for i in range(mask2.shape[0]):
        for j in range(mask2.shape[1]):
            if mask2[i,j] != 0:
                back_[i,j,:] = fore[i,j,:]
    back[h-size[1]:,w-size[0]:,:] = back_
    return back


def take_instruction(img_id,img_style,instruction):
    if instruction == 2:
        if img_id != None:
            img_id = (img_id+1)%3
        else:
            img_id = 0
    elif instruction == 3:
        if img_id != None:
            img_id = (img_id-1+3)%3
        else:
            img_id = 0
    elif instruction == 4:
        img_style = (img_style+1)%3
    else:
        img_style = (img_style-1+3)%3

    return img_id, img_style

def get_background(img_id, img_style):
    if img_style == 0:
        name = "./background/"+str(img_id)+"-"+str(img_style)+".jpg"
    else:
        name = "./background/"+str(img_id)+"-"+str(img_style)+".png"
    background = cv2.imread(name)
    return background


event_record = []
cap = cv2.VideoCapture(0)
ret,frame = cap.read()
image = Image.fromarray(frame)
#size = image.size
size = (1400, 1000)
roi = _get_roi(size, ratio = 0.5)
img_id = None
img_style = 0
while True:
    ret,frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.bilateralFilter(frame, 5, 50, 100)
    image = Image.fromarray(frame)
    #print image.size # 640 * 480
    array = np.asarray(image)
    array = mount_roi(array, roi, color = (74, 20, 140), thickness = 2)
    cv2.imshow('show', array)
    crop  = _crop_array(array, roi)
    event = grdetect(crop)
    mask = np.zeros(frame.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    rect = (150, 50,400,400)
    cv2.grabCut(frame,mask,rect,bgdModel,fgdModel,1,cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = frame*mask2[:,:,np.newaxis]
    for i in range(mask2.shape[0]):
        for j in range(mask2.shape[1]):
            if mask2[i,j] == 0:
                img[i,j,:] = [255, 255, 255]
    cv2.imshow("frame", img)
    if event != None and event !=0:
        event_record.append(event)
    if len(event_record) == 3:
        instruction = get_instruction(event_record)
        if instruction != None:
            print instruction
            event_record = []
            img_id, img_style = take_instruction(img_id,img_style,instruction)
            background = get_background(img_id, img_style)
        else:
            del event_record[0]
    if img_id != None:
        result = combine(frame, background, mask2)
        cv2.imshow("result", result)
    cv2.waitKey(100)      
cv2.destroyWindow()

