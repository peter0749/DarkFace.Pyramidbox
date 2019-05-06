#-*- coding:utf-8 -*-
import torch
import cv2
from PIL import Image, ImageDraw
import torch.utils.data as data
import numpy as np
import random
from data.config import cfg
from utils.augmentations import preprocess, spaaug
import matplotlib.pyplot as plt


class WIDERDetection(data.Dataset):
    """docstring for WIDERDetection"""

    def __init__(self, list_file, mode='train'):
        super(WIDERDetection, self).__init__()
        self.mode = mode
        self.fnames = []
        self.boxes = []

        with open(list_file) as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip().split()
            num_faces = int(line[1])
            box = []
            for i in range(num_faces):
                x = int(line[2 + 5 * i])
                y = int(line[3 + 5 * i])
                w = int(line[4 + 5 * i])
                h = int(line[5 + 5 * i])
                if w <= 0 or h <= 0:
                    continue
                box.append([x, y, x + w, y + h])
            if len(box) > 0:
                self.fnames.append(line[0])
                self.boxes.append(box)

        self.num_samples = len(self.boxes)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        img, face_target,head_target = self.pull_item(index)
        return img, face_target,head_target

    def pull_item(self, index):
        while True:
            image_path = self.fnames[index]
            img = np.asarray(cv2.imread(image_path, cv2.IMREAD_COLOR), dtype=np.uint8)[...,::-1] # BGR -> RGB
            boxes = np.array(self.boxes[index], dtype=np.int32)
            if self.mode=='train' and cfg.apply_distort:
                auged_d = spaaug(image=img, bboxes=boxes, category_id=[0]*len(boxes))
                img, boxes = auged_d['image'], np.asarray(auged_d['bboxes'], dtype=np.float32)
            im_height, im_width = img.shape[:2]
            if len(boxes)>0: # if has bounding boxes
                boxes = np.round(np.asarray(boxes, dtype=np.float32)).astype(np.int32)
                boxes[:,[0,2]] = boxes[:,[0,2]].clip(0, im_width-1)
                boxes[:,[1,3]] = boxes[:,[1,3]].clip(0, im_height-1)
                boxes = boxes[(boxes[:,0]<boxes[:,2]) & (boxes[:,1]<boxes[:,3])] # delete invalid bbox
                boxes = boxes.astype(np.float32)
            if len(boxes)>0:
                boxes = self.annotransform(boxes, im_width, im_height)
                bbox_labels = np.pad(boxes, ((0,0),(1,0)), 'constant', constant_values=1.0).tolist()
                img, sample_labels = preprocess(
                    img, bbox_labels, self.mode, image_path)
            else:
                sample_labels = []
            sample_labels = np.array(sample_labels)
            if len(sample_labels) > 0:
                face_target = np.hstack(
                    (sample_labels[:, 1:], sample_labels[:, 0][:, np.newaxis]))
                assert (face_target[:, 2] > face_target[:, 0]).all()
                assert (face_target[:, 3] > face_target[:, 1]).all()

                #img = img.astype(np.float32)
                face_box = face_target[:, :-1]
                head_box = self.expand_bboxes(face_box)
                head_target = np.hstack((head_box, face_target[
                                        :, -1][:, np.newaxis]))
                break
            else:
                index = random.randrange(0, self.num_samples)


        img_torch = torch.from_numpy(img)
        '''
        img = img[::-1]
        img += cfg.img_mean
        img = img[::-1]
        img = np.transpose(img, (1,2,0)).astype(np.uint8)
        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)
        w,h = img.size
        for bbox in sample_labels:
            bbox = (bbox[1:] * np.array([w, h, w, h])).tolist()
            draw.rectangle(bbox,outline='red')
        img.save('{:d}.jpg'.format(index))
        '''
        return img_torch, face_target, head_target


    def annotransform(self, boxes, im_width, im_height):
        boxes[:, 0] /= im_width
        boxes[:, 1] /= im_height
        boxes[:, 2] /= im_width
        boxes[:, 3] /= im_height
        return boxes

    def expand_bboxes(self,
                      bboxes,
                      expand_left=2.,
                      expand_up=2.,
                      expand_right=2.,
                      expand_down=2.):
        expand_bboxes = []
        for bbox in bboxes:
            xmin = bbox[0]
            ymin = bbox[1]
            xmax = bbox[2]
            ymax = bbox[3]
            w = xmax - xmin
            h = ymax - ymin
            ex_xmin = max(xmin - w / expand_left, 0.)
            ex_ymin = max(ymin - h / expand_up, 0.)
            ex_xmax = max(xmax + w / expand_right, 0.)
            ex_ymax = max(ymax + h / expand_down, 0.)
            expand_bboxes.append([ex_xmin, ex_ymin, ex_xmax, ex_ymax])
        expand_bboxes = np.array(expand_bboxes)
        return expand_bboxes

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    face_targets = []
    head_targets = []

    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        face_targets.append(torch.FloatTensor(sample[1]))
        head_targets.append(torch.FloatTensor(sample[2]))
    return torch.stack(imgs, 0), face_targets,head_targets

if __name__ == '__main__':
    dataset = WIDERDetection(cfg.FACE.TRAIN_FILE)
    for i in range(10):
        img, boxes = dataset[i][:2]
        #print(boxes)
        img = img.numpy()
        img = img[::-1]
        img += cfg.img_mean
        img = img[::-1]
        img = np.transpose(img, (1,2,0)).astype(np.uint8)
        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)
        w,h = img.size
        for bbox in boxes:
            bbox = (bbox[:-1] * np.array([w, h, w, h])).tolist()
            draw.rectangle(bbox,outline='red')
        img.save('{:d}.jpg'.format(i))
