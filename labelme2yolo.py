import cv2
import os
import json
import shutil
import numpy as np
from pathlib import Path
from glob import glob

id2cls = {0: 'aeroplane', 1: 'elevator car', 2: 'plane refueller', 3: 'tractor', 4: 'catering truck'}  # 换成自己数据集对应的类别
cls2id = {'aeroplane': 0, 'elevator car': 1, 'plane refueller': 2, 'tractor': 3, 'catering truck': 4} # 同上


# 支持中文路径
def cv_imread(filePath):
    cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), flags=cv2.IMREAD_COLOR)
    return cv_img


def labelme2yolo_single(imgs_path, label_file):
    anno = json.load(open(label_file, "r", encoding="utf-8"))
    shapes = anno['shapes']
    w0, h0 = anno['imageWidth'], anno['imageHeight']
    image_path = os.path.basename(imgs_path + anno['imagePath'])
    labels = []
    for s in shapes:
        pts = s['points']
        x1, y1 = pts[0][0], pts[0][1]
        x2, y2 = pts[1][0], pts[1][1]
        x = (x1 + x2) / (2.0 * w0)
        y = (y1 + y2) / (2.0 * h0)
        w = abs(x2 - x1) / (1.0 * w0)
        h = abs(y2 - y1) / (1.0 * h0)
        cid = cls2id[s['label']]
        labels.append([cid, x, y, w, h])
    return np.array(labels), image_path


def labelme2yolo(imgs_path, labelme_label_dir, txt_dir='res/'):
    labelme_label_dir = str(Path(labelme_label_dir)) + '/'
    txt_dir = str(Path(txt_dir))
    yolo_label_dir = txt_dir + '/'

    if not os.path.exists(yolo_label_dir):
        os.makedirs(yolo_label_dir)

    json_files = glob(labelme_label_dir + '*.json')
    for ijf, jf in enumerate(json_files):
        filename = os.path.basename(jf).rsplit('.', 1)[0]
        labels, image_path = labelme2yolo_single(imgs_path, jf)
        if len(labels) > 0:
            np.savetxt(yolo_label_dir + filename + '.txt', labels)
    print('Completed!')


if __name__ == '__main__':
    img_path = r'I:\YOLOX\datasets\coco\val2017'  # 数据集图片的路径---需要修改
    json_dir = r'I:\YOLOX\datasets\yolo\val_labelme'  # labelme标注的.json文件的路径---与上面的图片是一一对应的---需要修改
    save_dir = r'I:\YOLOX\datasets\yolo\val_txt_label'  # 保存转化后.txt标签的路径---需要修改
    labelme2yolo(img_path, json_dir, save_dir)

