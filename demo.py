#-*- coding:utf-8 -*-
import os
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable

import cv2
import time
import numpy as np
import numba as nb
from PIL import Image
from tqdm import tqdm

from data.config import cfg
from pyramidbox import build_net
from utils.augmentations import to_chw_bgr

parser = argparse.ArgumentParser(description='pyramidbox demo')
parser.add_argument('--img_path',
                    type=str, required=True,
                    help='Directory to images')
parser.add_argument('--save_dir',
                    type=str, default='',
                    help='Directory for detect result')
parser.add_argument('--save_pred',
                    type=str, default='pred_labels/',
                    help='Directory for detect result')
parser.add_argument('--save_format',
                    type=str, default='darkface',
                    help='darkface|map')
parser.add_argument('--model',
                    type=str, default='weights/pyramidbox.pth',
                    help='trained model')
parser.add_argument('--thresh',
                    default=0.4, type=float,
                    help='Final confidence threshold')
args = parser.parse_args()

if args.save_format!='darkface' and args.save_format!='map':
    raise ValueError('Unknown output format.')

if len(args.save_dir)>0:
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
if not os.path.exists(args.save_pred):
    os.makedirs(args.save_pred)

use_cuda = torch.cuda.is_available()

if use_cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

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

@nb.jit(fastmath=True)
def bbox_vote(det):
    order = det[:, 4].ravel().argsort()[::-1]
    det = det[order, :]
    dets = []
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
        mask = (o>=0.3)
        merge_index = np.where(mask)[0]
        reserve_index = np.where(~mask)[0]
        det_accu = det[merge_index, :]
        det = det[reserve_index]

        if merge_index.shape[0] <= 1:
            continue
        det_accu[:, 0:4] = det_accu[:, 0:4] * det_accu[:, -1:]
        max_score = np.max(det_accu[:, 4])
        det_accu_sum = np.zeros((1, 5))
        det_accu_sum[:, 0:4] = np.sum(
            det_accu[:, 0:4], axis=0) / np.sum(det_accu[:, -1:])
        det_accu_sum[:, 4] = max_score
        if len(dets)==0:
            dets = det_accu_sum
        else:
            dets = np.concatenate((dets, det_accu_sum))
    return dets

def detect(net, img_path):
    img = np.asarray(cv2.imread(img_path, cv2.IMREAD_COLOR))[...,::-1] # BGR -> RGB
    height, width, _ = img.shape
    max_im_shrink = np.sqrt(
        1700 * 1000 / (img.shape[0] * img.shape[1]))
    shrink = max_im_shrink if max_im_shrink < 1 else 1

    det0 = detect_face(net, img, shrink)
    det1 = flip_test(net, img, shrink)    # flip test
    det2, det3 = multi_scale_test(net, img, max_im_shrink)
    det = np.row_stack((det0, det1, det2, det3))
    dets = bbox_vote(det)

    with open(args.save_pred + '/' + os.path.splitext(os.path.split(img_path)[-1])[0] + '.txt', 'w') as fp:
        #if args.save_format=='darkface':
        #    fp.write('{:d}\n'.format(len(dets)))
        for bbox in dets:
            score = float(bbox[-1])
            xmin,ymin,xmax,ymax = map(lambda x: int(np.round(x)), bbox[:-1])
            xmin,xmax = (xmin,xmax) if xmin<xmax else (xmax,xmin)
            ymin,ymax = (ymin,ymax) if ymin<ymax else (ymax,ymin)
            if args.save_format=='map':
                fp.write('face {:.6f} {:d} {:d} {:d} {:d}\n'.format(score,xmin,ymin,xmax,ymax))
            else:
                fp.write('{:d} {:d} {:d} {:d} {:.6f}\n'.format(xmin,ymin,xmax,ymax,score))

    if len(args.save_dir)>0:
        img = img[...,::-1] # RGB -> BGR
        for bbox in dets:
            score = bbox[-1]
            left_up, right_bottom = tuple(map(int, bbox[:2])), tuple(map(int, bbox[2:4]))
            if left_up[0]>=right_bottom[0] or left_up[1]>=right_bottom[1]:
                continue
            img = img.copy()
            cv2.rectangle(img, left_up, right_bottom, (0, 0, 255), 2)
            conf = "{:.2f}".format(score)
            text_size, baseline = cv2.getTextSize(
                conf, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
            p1 = (left_up[0], left_up[1] - text_size[1])
            cv2.rectangle(img, (p1[0] - 2 // 2, p1[1] - 2 - baseline),
                          (p1[0] + text_size[0], p1[1] + text_size[1]), [255, 0, 0], -1)
            cv2.putText(img, conf, (p1[0], p1[
                        1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, 8)
        cv2.imwrite(os.path.join(args.save_dir, os.path.basename(img_path)), img)


if __name__ == '__main__':
    net = build_net('test', cfg.NUM_CLASSES)
    net.load_state_dict(torch.load(args.model, map_location="cuda" if use_cuda else "cpu"))
    net.eval()

    if use_cuda:
        net.cuda()
        torch.backends.benchmark = True

    img_path = args.img_path
    img_list = [os.path.join(img_path, x)
            for x in os.listdir(img_path) if x.endswith('png') ]
    for path in tqdm(img_list, total=len(img_list)):
        detect(net, path)
