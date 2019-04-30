#-*- coding:utf-8 -*-
import os
from data.config import cfg

def process_by_id(anno_pth):
    bboxes = []
    with open(anno_pth, "r") as afp:
        _ = afp.readline()
        for line in afp.readlines():
            xmin, ymin, xmax, ymax = tuple(map(int, line.split(' ')))
            bboxes += [[xmin,ymin,xmax-xmin,ymax-ymin]]
    return bboxes

def darkface_mklist(img_root, label_root, train=4500, val=500, test=1000):
    seq = [
            (cfg.FACE.TRAIN_FILE, list(range(1,train+1))),
            (cfg.FACE.VAL_FILE, list(range(train+1,train+val+1))),
            (cfg.FACE.TEST_FILE, list(range(train+val+1, train+val+test+1))),
          ]
    for (out, r) in seq:
        with open(out, "w") as fp:
            for id_ in r:
                img_pth = img_root+"/%d.png"%id_
                bboxes = process_by_id(label_root+"/%d.txt"%id_)
                fp.write("{} {:d}".format(img_pth, len(bboxes)))
                for (x,y,w,h) in bboxes:
                    fp.write(" {:d} {:d} {:d} {:d} 1".format(x,y,w,h))
                fp.write("\n")
if __name__ == '__main__':
    root = "/home/peter/NTU-MiRA/FR/DarkFace_Train"
    darkface_mklist(root+"/images", root+"/label", train=4500, val=500, test=1000)

