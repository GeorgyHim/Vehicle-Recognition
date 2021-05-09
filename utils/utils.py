import numpy as np
import cv2

_CLASS_NAMES_FILE = './data/coco.names'


def load_class_names():
    """
    Returns a list of string corresonding to class names and it's length
    """
    with open(_CLASS_NAMES_FILE, 'r') as f:
        class_names = f.read().splitlines()
    return class_names, len(class_names)


def draw_boxes_frame(frame, frame_size, boxes_dicts, class_names, input_size):
    """
    Draws detected boxes in a video frame
    """
    boxes_dict = boxes_dicts[0]
    resize_factor = (frame_size[0] / input_size[1], frame_size[1] / input_size[0])
    for cls in range(len(class_names)):
        boxes = boxes_dict[cls]
        color = (0, 0, 255)
        if np.size(boxes) != 0:
            for box in boxes:
                xy = box[:4]
                xy = [int(xy[i] * resize_factor[i % 2]) for i in range(4)]
                cv2.rectangle(frame, (xy[0], xy[1]), (xy[2], xy[3]), color[::-1], 2)
                (test_width, text_height), baseline = cv2.getTextSize(
                    class_names[cls], cv2.FONT_HERSHEY_SIMPLEX, 0.75, 1
                )
                cv2.rectangle(
                    frame,
                    (xy[0], xy[1]),
                    (xy[0] + test_width, xy[1] - text_height - baseline),
                    color[::-1],
                    thickness=cv2.FILLED
                )
                cv2.putText(
                    frame,
                    class_names[cls],
                    (xy[0], xy[1] - baseline),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 0, 0),
                    1
                )
