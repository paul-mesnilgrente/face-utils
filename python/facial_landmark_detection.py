# import the necessary packages
from collections import OrderedDict
import face_detection
import argparse
import numpy as np
import imutils
import dlib
import cv2

# define a dictionary that maps the indexes of the facial
# landmarks to specific face regions
FACIAL_LANDMARKS_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 36)),
    ("jaw", (0, 17))
])

def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords

class facial_landmark_detector:
    def __init__(self, model):
        self.predictor = dlib.shape_predictor(model)

    def predict(self, image, boxes):
        gray = imutils.resize(image, width=500)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        (h0, w0) = image.shape[:2]
        (h1, w1) = gray.shape[:2]

        shapes = []
        boxes = np.array(boxes * np.array([w1, h1, w1, h1]) / np.array([w0, h0, w0, h0]), dtype=int)
        # loop over the face detections
        for (i, rect) in enumerate(boxes):
            # convert my rect in dlib rectangle
            rect = dlib.rectangle(rect[0], rect[1], rect[2], rect[3])
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = self.predictor(gray, rect)
            shape = shape_to_np(shape)
            shape = np.array(shape * np.array([w0, h0]) / np.array([w1, h1]), dtype=int)
            shapes.append(shape)

        return shapes

    def draw(self, image, shapes):
        res = image.copy()
        for shape in shapes:
            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image
            for (x, y) in shape:
                cv2.circle(res, (x, y), 1, (0, 0, 255), -1)
        return res


if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prototxt",
        default="../models/deploy.prototxt",
        help="path to facial landmark predictor")
    ap.add_argument("-m", "--model",
        default="../models/res10_300x300_ssd_iter_140000_fp16.caffemodel",
        help="path to facial landmark predictor")
    ap.add_argument("-s", "--shape-predictor",
        default="../models/shape_predictor_68_face_landmarks.dat",
        help="path to facial landmark predictor")
    args = vars(ap.parse_args())
    print(args)

    # initialize the detectors
    detector = face_detection.face_detector(args["prototxt"], args["model"])
    predictor = facial_landmark_detector(args["shape_predictor"])

    cap = cv2.VideoCapture(0)

    while cv2.waitKey(1) != 27:
        ret, frame = cap.read()
        (boxes, confidences) = detector.detect(frame)
        shapes = predictor.predict(frame, boxes)

        frame = detector.draw(frame, boxes, confidences)
        frame = predictor.draw(frame, shapes)
        cv2.imshow("Camera", frame)