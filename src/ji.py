from __future__ import print_function

import logging as log
import json
import os
import cv2
import datetime
import numpy as np
import argparse

from utils.datasets import *
from utils.general import *

log.basicConfig(level=log.DEBUG)

device = torch.device('cuda:0')
half = device.type != 'cpu'  # half precision only supported on CUDA
half=False
    
class opt:
        weights = "/usr/local/ev_sdk/model/best.pt"
        # weights = "/project/train/models/final/best.pt"
        img_size = 512
        conf_thres = 0.5
        iou_thres = 0.6
        augment = True
        device = 'cuda:0'
        classes=1
        agnostic_nms = True

def init():
    """Initialize model

    Returns: model

    """
    weights, imgsz = opt.weights,opt.img_size
    source = '../input/global-wheat-detection/test/'
    
    # Initialize
    device = torch.device('cuda:0')
    # Load model

    model = torch.load(weights, map_location=device)['model'].to(device).float().eval()
    
    
    if half:
        model.half()
    return model


def process_image(model, img, args=None):
    """Do inference to analysis input_image and get output

    Attributes:
        net: model handle
        input_image (numpy.ndarray): image to be process, format: (h, w, c), BGR
        args: optional args

    Returns: process result

    """

    # ------------------------------- Prepare input -------------------------------------
    if not model or img is None:
        log.error('Invalid input args')
        return None
    ih, iw, _ = img.shape

    device = "cuda:0"
    
    t1 = datetime.datetime.now()
    
    # letter image
    im0_shape = img.shape
    img = letterbox(img, new_shape=opt.img_size)[0]
    # print(img0_shape, img.shape)
    img = img.astype(np.float32)
    
    imgsz = opt.img_size
    # img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    img = torch.from_numpy(img).to(device)
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    img = img.permute(2, 0, 1).contiguous()
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    
        
    bboxes_2 = []
    score_2 = []
    
    
    if half:
        img = img.half()

    t2 = datetime.datetime.now()
    
    pred = model(img, augment=opt.augment)[0]
    
    t3 = datetime.datetime.now()
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,merge=False, classes=None, agnostic=False)[0]

    # --------------------------- Read and postprocess output ---------------------------
    t4 = datetime.datetime.now()
    
    print('image letter:{}'.format(t2-t1))
    print('inference:{}'.format(t3-t2))
    print("NMS:{}".format(t4-t3))
    if pred==None:
        detect_objs = []
        return json.dumps({"objects": detect_objs})
    
    # pred = pred.cpu()
    # boxes = pred[:, :4]
    

    """
    boxes[:, 2] += boxes[:, 0]
    boxes[:, 3] += boxes[:, 1]
    boxes = np.array(boxes).astype(np.int32)
    """
    
    pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], im0_shape).round()
    
    scores = pred[:, 4]

    detect_objs = []
    # for k, score in enumerate(scores):
    for *xyxy, conf, cls in pred:
        xmin, ymin, xmax, ymax = xyxy
        detect_objs.append({
            'name': "vendors",
            'xmin': int(xmin),
            'ymin': int(ymin),
            'xmax': int(xmax),
            'ymax': int(ymax)
        })
    return json.dumps({"objects": detect_objs})
    

if __name__ == '__main__':
    """Test python api
    """
    
    img = cv2.imread('/home/data/51/vendor20200908_2123.jpg')
    
    predictor = init()

    begin = datetime.datetime.now()

    result = process_image(predictor, img)
    end = datetime.datetime.now()
    print("total time:{}".format(end-begin))
    log.info(result)