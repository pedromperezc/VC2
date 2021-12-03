import pandas as pd
import cv2
import numpy as np
from matplotlib import pyplot as plt
import json
from sklearn.model_selection import train_test_split

df = pd.read_csv("train.csv")
grouped_df = df.groupby("ImageId")
grouped_ClassId = grouped_df["ClassId"].apply(list)
grouped_EncodedPixels = grouped_df["EncodedPixels"].apply(list)
grouped_ClassId = grouped_ClassId.reset_index()
grouped_EncodedPixels = grouped_EncodedPixels.reset_index()

df_final = grouped_ClassId.merge(grouped_EncodedPixels, left_on='ImageId', right_on='ImageId')

train_df, test_df = train_test_split(df_final, test_size=0.2, random_state=69)


def rle_to_mask(rle, width=1600, height=256):
    """ Transform run-length encoding string to mask array """

    # Get all rle elements from rle string:
    rle_list = rle.split()

    # Convert all elements into integers:
    rle_integers = [int(x) for x in rle_list]

    # Create pairs in previous list:
    rle_pairs = np.array(rle_integers).reshape(-1, 2)

    # Initialize mask array:
    mask_array = np.zeros(width * height, dtype=np.uint8)

    # Populate mask array:
    for index, length in rle_pairs:
        index -= 1
        mask_array[index:index + length] = 1

    # Reshape mask array:
    mask = mask_array.reshape(width, height).T

    # Return result:
    return mask


coco = {
    "info": {
        "year": "2021",
        "version": "1",
        "description": "Steel Defect Detection",
        "contributor": "Severstal",
        "url": "https://www.severstal.com/eng/",
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

for i in range(len(train_df)):
    print(i)
    img = cv2.imread("train/" + train_df.iloc[i].ImageId)
    coco["images"].append({"id": i,
                           "license": 1,
                           "file_name": train_df.iloc[i].ImageId,
                           "height": img.shape[0],
                           "width": img.shape[1]
                           })

    for j in range(len(train_df.iloc[i].ClassId)):

        mask = rle_to_mask(train_df.iloc[i].EncodedPixels[j])
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # for p, contour in enumerate(contours):
        #     mask_layer = np.reshape(contour, (2 * contour.shape[0],))
        #     x, y, w, h = cv2.boundingRect(contour)
        #
        #     # Se agrega esto por un bug, si la segmentación solo tiene 4 puntos da error en la validación
        #     if len(mask_layer) == 4:
        #         mask_layer = np.append(mask_layer, mask_layer[0:2])
        # coco["annotations"].append({
        #     "id": str(i) + "_" + str(j) + "_" + str(p),
        #     "area": h * w,
        #     "image_id": i,
        #     "category_id": int(train_df.iloc[i].ClassId[j]),
        #     "bbox": [x, y, w, h],
        #     "segmentation": [[float(x) for x in mask_layer.tolist()]],
        #     "iscrowd": 0
        # })

        for p, contour in enumerate(contours):
            contour_array = contour.reshape(-1, 2)
            px = [a[0] for a in contour_array]
            py = [b[1] for b in contour_array]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]
            x0, y0 = int(np.min(px)), int(np.min(py))
            x1, y1 = int(np.max(px)), int(np.max(py))
            if (len(poly) % 2) == 0 and len(poly) >= 6:
                coco["annotations"].append({
                    "id": str(i) + "_" + str(j) + "_" + str(p),
                    "area": abs(x0 - x1) * abs(y0 - y1),
                    "image_id": i,
                    "category_id": int(train_df.iloc[i].ClassId[j]),
                    "bbox": [x0, y0, abs(x0 - x1), abs(y0 - y1)],
                    "segmentation": [poly],
                    "iscrowd": 0
                })

with open('train_coco_dataset.json', 'w') as f:
    json.dump(coco, f)



coco = {
    "info": {
        "year": "2021",
        "version": "1",
        "description": "Steel Defect Detection",
        "contributor": "Severstal",
        "url": "https://www.severstal.com/eng/",
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

for i in range(len(test_df)):
    print(i)
    img = cv2.imread("train/" + test_df.iloc[i].ImageId)
    coco["images"].append({"id": i,
                           "license": 1,
                           "file_name": test_df.iloc[i].ImageId,
                           "height": img.shape[0],
                           "width": img.shape[1]
                           })

    for j in range(len(test_df.iloc[i].ClassId)):

        mask = rle_to_mask(test_df.iloc[i].EncodedPixels[j])
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # for p, contour in enumerate(contours):
        #     mask_layer = np.reshape(contour, (2 * contour.shape[0],))
        #     x, y, w, h = cv2.boundingRect(contour)
        #
        #     # Se agrega esto por un bug, si la segmentación solo tiene 4 puntos da error en la validación
        #     if len(mask_layer) == 4:
        #         mask_layer = np.append(mask_layer, mask_layer[0:2])
        # coco["annotations"].append({
        #     "id": str(i) + "_" + str(j) + "_" + str(p),
        #     "area": h * w,
        #     "image_id": i,
        #     "category_id": int(train_df.iloc[i].ClassId[j]),
        #     "bbox": [x, y, w, h],
        #     "segmentation": [[float(x) for x in mask_layer.tolist()]],
        #     "iscrowd": 0
        # })

        for p, contour in enumerate(contours):
            contour_array = contour.reshape(-1, 2)
            px = [a[0] for a in contour_array]
            py = [b[1] for b in contour_array]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]
            x0, y0 = int(np.min(px)), int(np.min(py))
            x1, y1 = int(np.max(px)), int(np.max(py))
            if (len(poly) % 2) == 0 and len(poly) >= 6:
                coco["annotations"].append({
                    "id": str(i) + "_" + str(j) + "_" + str(p),
                    "area": abs(x0 - x1) * abs(y0 - y1),
                    "image_id": i,
                    "category_id": int(test_df.iloc[i].ClassId[j]),
                    "bbox": [x0, y0, abs(x0 - x1), abs(y0 - y1)],
                    "segmentation": [poly],
                    "iscrowd": 0
                })

with open('test_coco_dataset.json', 'w') as f:
    json.dump(coco, f)