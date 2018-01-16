import torch
import numpy as np
import argparse
from matplotlib import pyplot as plt
import xml.etree.ElementTree as ET
import cv2

colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()

parser = argparse.ArgumentParser(description='Visualize ground truth boxes on a image.')
parser.add_argument('--img', help='Image path relative to VID data root',
                    default='train/ILSVRC2015_VID_train_0000/ILSVRC2015_train_00043000/000085')
args = parser.parse_args()


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text) - 1,
                              int(bbox.find('ymin').text) - 1,
                              int(bbox.find('xmax').text) - 1,
                              int(bbox.find('ymax').text) - 1]
        objects.append(obj_struct)

    return objects


if __name__=='__main__':
    image_path = '/home/leoyolo/data/ILSVRC/Data/VID/{}.JPEG'.format(args.img)
    gt_path = '/home/leoyolo/data/ILSVRC/Annotations/VID/{}.xml'.format(args.img)
    gt = parse_rec(gt_path)
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 10))
    plt.imshow(img)  # plot the image for matplotlib
    currentAxis = plt.gca()
    for i in range(len(gt)):
        label_name = gt[i]['name']
        display_txt = label_name
        pt = gt[i]['bbox']
        coords = (pt[0], pt[1]), pt[2] - pt[0] + 1, pt[3] - pt[1] + 1
        color = colors[i]
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor': color, 'alpha': 0.5})


    plt.show()