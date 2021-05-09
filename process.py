import argparse

from convert_weights import convert_weights
from detect import detect

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help='Path to video file')
    parser.add_argument('--log', action='store_true', help='Enable timing log')
    parser.add_argument('--iou', type=float, default=0.5, help='IoU threshold')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold')

    args = parser.parse_args()
    convert_weights()
    detect(args.path, args.log, args.iou, args.confidence)
