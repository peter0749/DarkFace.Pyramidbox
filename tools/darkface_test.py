#-*- coding:utf-8 -*-
import os
import torch
import argparse
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import os.path as osp

import cv2
import time
from tqdm import tqdm
import numpy as np
import numba as nb
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data.config import cfg
from pyramidbox import build_net
from torch.autograd import Variable
from utils.augmentations import to_chw_bgr


parser = argparse.ArgumentParser(description='pyramidbox evaluatuon wider')
parser.add_argument('--model', type=str,
                    default='weights/pyramidbox.pth', help='trained model')
parser.add_argument('--thresh', default=0.05, type=float,
                    help='Final confidence threshold')
parser.add_argument('--iou', default=0.5, type=float,
                    help='')
args = parser.parse_args()


use_cuda = torch.cuda.is_available()

if use_cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

@nb.jit(nopython=True)
def IOU(y_true, y_pred):
    xmin1, ymin1, xmax1, ymax1 = y_true
    xmin2, ymin2, xmax2, ymax2 = y_pred
    inner = max(0,(min(xmax1,xmax2)-max(xmin1,xmin2)) * (min(ymax1,ymax2)-max(ymin1,ymin2)))
    area1 = max(0,(xmax1-xmin1)*(ymax1-ymin1))
    area2 = max(0,(xmax2-xmin2)*(ymax2-ymin2))
    return inner / (area1+area2-inner) # intersection / union

def get_gt():
    res = {}
    with open(cfg.FACE.TEST_FILE, 'r') as fp:
        for line in fp.readlines():
            l = line.strip('\n').split(' ')
            path, n = l[:2]
            n = int(n)
            bboxes = []
            for i in range(n):
                x,y,w,h = tuple(map(int, l[ i*5 + 2 : i*5 + 4 + 2])) # (path, N, x1,y1,w1,h1,s1, x2,y2,w2,h2,s2, ...)
                bboxes += [(x,y,x+w,y+h)]
            if path not in res:
                res[path] = bboxes
            else:
                res[path] += bboxes
    return res

def detect_face(net, img, shrink):
    if shrink != 1:
        img = cv2.resize(img, None, None, fx=shrink, fy=shrink,
                         interpolation=cv2.INTER_LINEAR)

    x = to_chw_bgr(img)
    x = x.astype('float32')
    x -= cfg.img_mean
    x = x[[2, 1, 0], :, :]

    with torch.no_grad():
        x = Variable(torch.from_numpy(x).unsqueeze(0))

        if use_cuda:
            x = x.cuda()
        # print(x.size())
        y = net(x)
        detections = y.detach()
        detections = detections.cpu().numpy()

    det_conf = detections[0, 1, :, 0]
    det_xmin = img.shape[1] * detections[0, 1, :, 1] / shrink
    det_ymin = img.shape[0] * detections[0, 1, :, 2] / shrink
    det_xmax = img.shape[1] * detections[0, 1, :, 3] / shrink
    det_ymax = img.shape[0] * detections[0, 1, :, 4] / shrink
    det = np.column_stack((det_xmin, det_ymin, det_xmax, det_ymax, det_conf))

    keep_index = np.where(det[:, 4] >= args.thresh)[0]
    det = det[keep_index, :]

    return det


def flip_test(net, image, shrink):
    image_f = cv2.flip(image, 1)
    det_f = detect_face(net, image_f, shrink)

    det_t = np.zeros(det_f.shape)
    det_t[:, 0] = image.shape[1] - det_f[:, 2]
    det_t[:, 1] = det_f[:, 1]
    det_t[:, 2] = image.shape[1] - det_f[:, 0]
    det_t[:, 3] = det_f[:, 3]
    det_t[:, 4] = det_f[:, 4]
    return det_t


def multi_scale_test(net, image, max_im_shrink):
    # shrink detecting and shrink only detect big face
    st = 0.5 if max_im_shrink >= 0.75 else 0.5 * max_im_shrink
    det_s = detect_face(net, image, st)
    index = np.where(np.maximum(
        det_s[:, 2] - det_s[:, 0] + 1, det_s[:, 3] - det_s[:, 1] + 1) > 30)[0]
    det_s = det_s[index, :]

    # enlarge one times
    bt = min(2, max_im_shrink) if max_im_shrink > 1 else (
        st + max_im_shrink) / 2
    det_b = detect_face(net, image, bt)

    # enlarge small image x times for small face
    if max_im_shrink > 2:
        bt *= 2
        while bt < max_im_shrink:
            det_b = np.row_stack((det_b, detect_face(net, image, bt)))
            bt *= 2
        det_b = np.row_stack((det_b, detect_face(net, image, max_im_shrink)))

    # enlarge only detect small face
    if bt > 1:
        index = np.where(np.minimum(
            det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1) < 100)[0]
        det_b = det_b[index, :]
    else:
        index = np.where(np.maximum(
            det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1) > 30)[0]
        det_b = det_b[index, :]

    return det_s, det_b

@nb.njit
def bbox_vote(det):
    dets = np.empty([])
    order = det[:, 4].ravel().argsort()[::-1]
    det = det[order, :]
    while det.shape[0] > 0:
        # IOU
        area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[0] + area[:] - inter)

        # get needed merge det and delete these det
        merge_index = np.where(o >= 0.3)[0]
        det_accu = det[merge_index, :]
        det = np.delete(det, merge_index, 0)

        if merge_index.shape[0] <= 1:
            continue
        det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
        max_score = np.max(det_accu[:, 4])
        det_accu_sum = np.zeros((1, 5))
        det_accu_sum[:, 0:4] = np.sum(
            det_accu[:, 0:4], axis=0) / np.sum(det_accu[:, -1:])
        det_accu_sum[:, 4] = max_score
        if dets.ndim==0:
            dets = det_accu_sum
        else:
            dets = np.row_stack((dets, det_accu_sum))

    dets = dets[0:750, :]
    return dets

if __name__ == '__main__':
    gt = get_gt()
    cfg.USE_NMS = False
    net = build_net('test', cfg.NUM_CLASSES)
    net.load_state_dict(torch.load(args.model, map_location="cuda" if use_cuda else "cpu"))
    net.eval()

    n_bins = 100
    thresholds = np.linspace(0.0,1.0,n_bins)

    tps = np.zeros(n_bins)
    #tns = np.zeros(n_bins)
    fns = np.zeros(n_bins)
    fps = np.zeros(n_bins)

    if use_cuda:
        net.cuda()
        torch.backends.cudnn.benchmark = True

    bbox_preds = []

    for img_path in tqdm(list(gt), total=len(list(gt))):
        #img = Image.open(img_path)
        #if img.mode == 'L':
        #    img = img.convert('RGB')
        #img = np.array(img)
        img = np.asarray(cv2.imread(img_path, cv2.IMREAD_COLOR))[...,::-1] # bgr -> rgb

        max_im_shrink = np.sqrt(
            848 * 480 / (img.shape[0] * img.shape[1]))

        shrink = max_im_shrink if max_im_shrink < 1 else 1

        det0 = detect_face(net, img, shrink)
        det1 = flip_test(net, img, shrink)    # flip test
        [det2, det3] = multi_scale_test(net, img, max_im_shrink)
        det = np.row_stack((det0, det1, det2, det3))
        dets = bbox_vote(det)

        bbox_preds.append(dets)

    for img_path, dets in tqdm(zip(list(gt),bbox_preds), total=len(list(gt))):
        for th_n,th in enumerate(thresholds):
            hit = np.zeros(len(gt[img_path]), dtype=np.bool)
            for bbox in dets:
                #xmin,ymin,xmax,ymax,score = bbox
                iou_max = 0
                idx = 0
                for i,bbox_y in enumerate(gt[img_path]):
                    iou = IOU(bbox_y, bbox[:-1])
                    if iou>iou_max:
                        iou_max = iou
                        idx = i
                if hit[idx]:
                    fps[th_n]+=1
                else:
                    hit[idx]=True
                    tps[th_n]+=1
            fns[th_n] += np.sum(~hit)

    precision = tps / (tps+fps)
    recall = tps / (tps+fns)
    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = ({'step': 'post'} if 'step' in signature(plt.fill_between).parameters else {})
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
    plt.savefig('PR_curve.png')

