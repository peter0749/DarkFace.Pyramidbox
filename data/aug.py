import albumentations as albu
from imgaug import augmenters as iaa
import cv2
import numpy as np

class Aug(object):
    def __init__(self):
        self.fast_aug = albu.Compose([
            albu.OneOf([
                albu.RandomBrightnessContrast(0.2, 0.2, p=0.5),
                albu.RandomGamma((78, 130),p=0.5),
                albu.CLAHE(4.0, (32,32), p=0.1) # we have big images
            ]),
            albu.JpegCompression(80,100,p=0.1),
            albu.OneOf([
                albu.Blur(5,p=1),
                albu.MedianBlur(5,p=1),
                albu.GaussianBlur(5,p=1)
            ],p=0.2),
            albu.GaussNoise((5,10),p=0.2),
            albu.OneOf([
                albu.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=30, p=1),
                albu.RGBShift(20,10,20,p=1),
            ], p=0.5),
            albu.ElasticTransform(alpha=5, sigma=50, alpha_affine=0, approximate=True, p=0.2)
        ], p=0.5)
        '''
        Bounding Box index: pixel based
        '''
        self.spaaug = albu.Compose([
            albu.Rotate(15, interpolation=cv2.INTER_LINEAR,\
            border_mode=cv2.BORDER_CONSTANT, p=1.0,\
            always_apply=True)],\
            bbox_params={'format': 'pascal_voc', 'min_area': 2, 'label_fields': ['category_id']}, p=0.5)

        # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
        # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
        sometimes = lambda aug: iaa.Sometimes(0.2, aug)
        self.seq = iaa.Sequential(
            [
                sometimes(iaa.MultiplyElementwise((0.5, 1.5), per_channel=0.5)),
                # execute 0 to 2 of the following (less important) augmenters per image
                # don't execute all of them, as that would often be way too strong
                iaa.SomeOf((0, 2),
                    [
                        sometimes(iaa.Superpixels(p_replace=(0, 0.1), n_segments=(2048, 4096))), # convert images into their superpixel representation
                        iaa.Sharpen(alpha=(0, 0.3), lightness=(0.75, 1.5)), # sharpen images
                        iaa.Emboss(alpha=(0, 0.3), strength=(0, 2.0)), # emboss images
                        # search either for all edges or for directed edges,
                        # blend the result with the original image using a blobby mask
                        sometimes(iaa.SimplexNoiseAlpha(iaa.OneOf([
                            iaa.EdgeDetect(alpha=(0.5, 1.0)),
                            iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                        ]))),
                        iaa.OneOf([
                            iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                            iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                        ]),
                        iaa.Grayscale(alpha=(0.0, 0.3)),
                    ],
                    random_order=True
                )
            ],
            random_order=True
        )

    def ultimate_aug(self, img, boxes):
        '''
            Input:
                img: np.array(dtype=np.uint8) [H,W,C{R,G,B}]
                boxes: np.array (pixel base) (xmin, ymin, xmax, ymax)
            Output:
                img: np.array(dtype=np.uint8)
                boxes: np.array (dtype=np.float32)
        '''
        img = self.fast_aug(image=img)['image']
        image_aug = self.seq.augment_image(img)
        auged_d = self.spaaug(image=image_aug, bboxes=boxes, category_id=[0]*len(boxes))
        img, boxes = auged_d['image'], np.asarray(auged_d['bboxes'], dtype=np.float32)

        return img,boxes

