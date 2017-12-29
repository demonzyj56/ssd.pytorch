""" Visualization script for ImageNetVID. """
import argparse
import cv2
import torch
from torch.autograd import Variable
import numpy as np
from data import v2_512, v2
from data import VID_CLASSES, VIDroot, VIDVideoDetection, BaseTransform_video
from ssd_video import build_ssd_video
from eval_vid_video import Timer

parser = argparse.ArgumentParser(description='Visualization for ImageNetVID')
parser.add_argument('--size', default=512, type=int, help='use 300 or 512 as base size')
parser.add_argument('--version', default='v2_512', help='conv11_2(v2) or pool6(v1) as last layer')
parser.add_argument('--pretrained_model', default=None, help='Load pretrained model to use')
parser.add_argument('--seed', default=-1, type=int, help='Random seed to set, negative values means not setting')
parser.add_argument('--K', default=0, help='The size of video context')
parser.add_argument('--no_cuda', action="store_true", default=False, help="Disable cuda")
parser.add_argument('--vid_val_list', default=None, help='Which validation list to visualize')
parser.add_argument('--thresh', default=0., type=float, help='Threshold for confidence values')
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
else:
    torch.set_default_tensor_type('torch.FloatTensor')

dataset_mean = (104, 117, 123)

def detect_single_frame(net, torch_img):
    """ Returns the detections on a single image. """
    x = Variable(torch_img)
    if args.cuda:
        x = x.cuda()
    return net(x).data


def visualize_detection(img, detections, thresh=0.):
    """ Visualizes the detection values returned from net.
    Here img is a numpy image not been resized. """
    h = img.shape[0]
    w = img.shape[1]
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
        # cls_dets = np.hstack((boxes.cpu().numpy(), scores[:, np.newaxis])) \
        #     .astype(np.float32, copy=False)
        for k in range(boxes.shape[0]):
            display_text = "{:s}: {:.2f}".format(VID_CLASSES[j-1], scores[k])
            x1, y1, x2, y2 = boxes[k, :]
            cv2.rectangle(img, boxes[k, :2], boxes[k, 2:], color=(0, 0, 0),
                          thickness=2)
            cv2.putText(img, display_text, boxes[k, :2], fontFace=0, fontScale=0.3,
                        color=(0, 0, 0))

    cv2.imshow("img", img)


if __name__ == "__main__":
    net = build_ssd_video("test", args.size, len(VID_CLASSES)+1, args.K)
    net.load_state_dict(torch.load(args.pretrained_model))
    net.eval()
    if args.cuda():
        net = net.cuda()
    print("Finished loading model {}".format(args.pretrained_model))
    # load data
    dataset = VIDVideoDetection([args.vid_val_list], 'data/', VIDroot,
                                transform=BaseTransform_video(args.size, dataset_mean),
                                is_test=True, k=args.K)
    timer = Timer()
    for i in (0,):
        img, _ = dataset[i]
        timer.tic()
        detections = detect_single_frame(net, img)
        detect_time = timer.toc(average=False)
        print("im_detect: {:d}/{:d} {:.3f}s".format(
            i + 1, len(dataset), detect_time
        ))
        display_img = cv2.imread(dataset.gt_roidb[i]['image'], cv2.IMREAD_COLOR)
        display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        visualize_detection(display_img, detections, thresh=args.thresh)