""" Visualization script for ImageNetVID. """
import argparse
import time
import os
import cv2
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import numpy as np
from data import VID_CLASSES, VIDroot, VIDVideoDetection, BaseTransform_video
from ssd_video import build_ssd_video

parser = argparse.ArgumentParser(description='Visualization for ImageNetVID')
parser.add_argument('--size', default=512, type=int, help='use 300 or 512 as base size')
parser.add_argument('--version', default='v2_512', help='conv11_2(v2) or pool6(v1) as last layer')
parser.add_argument('--pretrained_model', default=None, help='Load pretrained model to use')
parser.add_argument('--seed', default=-1, type=int, help='Random seed to set, negative values means not setting')
parser.add_argument('--K', default=0, type=int, help='The size of video context')
parser.add_argument('--no_cuda', action="store_true", default=False, help="Disable cuda")
parser.add_argument('--vid_val_list', default=None, help='Which validation list to visualize')
parser.add_argument('--thresh', default=0., type=float, help='Threshold for confidence values')
parser.add_argument('--save_folder', default='data/visualization', help='Root folder to save visualization results')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Ensure determinism
if args.seed >= 0:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.backends.cudnn.benchmark = True
else:
    torch.set_default_tensor_type('torch.FloatTensor')

dataset_mean = (104, 117, 123)
colors = plt.cm.hsv(np.linspace(0, 1, len(VID_CLASSES))).tolist()


def detect_single_frame(net, torch_img):
    """ Returns the detections on a single image. """
    x = Variable(torch_img)
    if args.cuda:
        x = x.cuda()
    return net(x).data


def visualize_detection(img, detections, thresh=0., save_filename=None):
    """ Visualizes the detection values returned from net.
    Here img is a numpy image not been resized. """
    h = img.shape[0]
    w = img.shape[1]
    # _, ax = plt.subplots(figsize=(12, 12))
    ax = plt.gca()
    ax.imshow(img, aspect='equal')
    # j = 0 denotes the background
    for j in range(1, detections.size(1)):
        dets = detections[0, j, :]
        mask = dets[:, 0].gt(thresh).expand(5, dets.size(0)).t()
        dets = torch.masked_select(dets, mask).view(-1, 5)
        if dets.dim() == 0:
            continue
        boxes = dets[:, 1:].cpu().numpy()
        boxes[:, 0] *= w
        boxes[:, 2] *= w
        boxes[:, 1] *= h
        boxes[:, 3] *= h
        scores = dets[:, 0].cpu().numpy()
        for k in range(boxes.shape[0]):
            display_text = "{:s}: {:.2f}".format(VID_CLASSES[j-1], scores[k])
            x1, y1, x2, y2 = boxes[k, :]
            ax.add_patch(
                plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False,
                              edgecolor=colors[j-1], linewidth=3.5)
            )
            ax.text(x1, y1-2, display_text, bbox=dict(facecolor=colors[j-1], alpha=0.5),
                    fontsize=14, color='white')
    plt.axis('off')
    plt.tight_layout()
    if save_filename is not None:
        plt.savefig(save_filename, bbox_inches='tight')
    else:
        plt.show()
    plt.clf()


if __name__ == "__main__":
    net = build_ssd_video("test", args.size, len(VID_CLASSES)+1, args.K)
    net.load_state_dict(torch.load(args.pretrained_model))
    net.eval()
    if args.cuda:
        net = net.cuda()
    print("Finished loading model {}".format(args.pretrained_model))
    # load data
    dataset = VIDVideoDetection([args.vid_val_list], 'data/', VIDroot,
                                transform=BaseTransform_video(args.size, dataset_mean),
                                is_test=True, k=args.K)
    save_folder = os.path.join(args.save_folder, args.vid_val_list)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    for _ in range(5):
        dummy = Variable(torch.randn(1, 3, args.size, args.size))
        if args.cuda:
            dummy = dummy.cuda()
        net(dummy)
    for i in range(len(dataset)):
        img, _ = dataset[i]
        img = Variable(img)
        if args.cuda:
            img = img.cuda()
        tic = time.time()
        detections = net(img).data
        detect_time = time.time() - tic
        display_img = cv2.imread(dataset.gt_roidb[i]['image'], cv2.IMREAD_COLOR)
        display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        tic = time.time()
        visualize_detection(display_img, detections, thresh=args.thresh,
                            save_filename=os.path.join(save_folder, '{}.jpg'.format(i)))
        visualize_time = time.time() - tic
        print("{:d}/{:d}, im_detect: {:.3f}s, visualize: {:.3f}s".format(
            i + 1, len(dataset), detect_time, visualize_time
        ))