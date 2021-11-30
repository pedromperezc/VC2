import pandas as pd
import cv2
import numpy as np
from matplotlib import pyplot as plt
import json

df = pd.read_csv('train.csv')


def rle_to_mask(lre, shape=(1600, 256)):
    '''
    params:  rle   - run-length encoding string (pairs of start & length of encoding)
             shape - (width,height) of numpy array to return

    returns: numpy array with dimensions of shape parameter
    '''
    # the incoming string is space-delimited
    runs = np.asarray([int(run) for run in lre.split(' ')])

    # we do the same operation with the even and uneven elements, but this time with addition
    runs[1::2] += runs[0::2]
    # pixel numbers start at 1, indexes start at 0
    runs -= 1

    # extract the starting and ending indeces at even and uneven intervals, respectively
    run_starts, run_ends = runs[0::2], runs[1::2]

    # build the mask
    h, w = shape
    mask = np.zeros(h * w, dtype=np.uint8)
    for start, end in zip(run_starts, run_ends):
        mask[start:end] = 1

    # transform the numpy array from flat to the original image shape
    return mask.reshape(shape)

coco ={
    "info": {
        "year": "2021",
        "version": "1",
        "description": "Steel Defect Detection",
        "contributor": "UBA",
        "url": "",
        "date_created": "2021-11-29T19:36:39+00:00"
    },
    "licenses": [
        {
            "id": 1,
            "url": "https://creativecommons.org/publicdomain/zero/1.0/",
            "name": "Public Domain"
        }
    ],
    "categories": [
        {
            "id": 0,
            "name": "class 0"
        },
        {
            "id": 1,
            "name": "class 1"},
        {
            "id": 2,
            "name": "class 2"},
        {
            "id": 3,
            "name": "class 3"},
        {
            "id": 4,
            "name": "class 4"}
    ],
    "images": [],
    "annotations": []

}


for i in range(len(df)):
  print(i)
  row = df.iloc[i]
  img = cv2.imread("train/"+ row.ImageId)
  coco["images"].append({"id": row.ImageId[:-4],
            "license": 1,
            "file_name": row.ImageId,
            "height": img.shape[0],
            "width": img.shape[1]
        })
  mask = rle_to_mask(row.EncodedPixels)
  contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  mask_layer = np.reshape(contours[0], (2 * contours[0].shape[0],))
  y, x, h, w = cv2.boundingRect(contours[0])


  # plt.imshow(img4)
  # plt.show()

  coco["annotations"].append({
            "id": i,
            "area": h*w,
            "image_id": row.ImageId[:-4],
            "category_id": int(row.ClassId),
            "bbox": [x,y,w,h],
            "segmentation": [mask_layer.tolist()],
            "iscrowd": 0
        })

with open('train_coco_dataset.json', 'w') as f:
    json.dump(coco, f)