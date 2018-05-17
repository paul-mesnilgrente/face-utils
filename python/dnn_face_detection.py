import argparse
import numpy as np
import cv2


class face_detector:
    def __init__(self, weights, model):
        self.network = cv2.dnn.readNetFromCaffe(weights, model)

    def detect(self, image, min_confidence=0.5):
        (h, w) = image.shape[:2]
        # grab the image dimensions and convert it to a blob
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the detections and
        # predictions
        self.network.setInput(blob)
        detections = self.network.forward()
        boxes = []
        confidences = []
        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence < min_confidence:
                continue

            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            confidences.append(confidence)
            boxes.append((startX, startY, endX, endY))
        return (boxes, confidences)

    def draw(self, image, boxes, confidences, min_confidence=0.5):
        res = image.copy()
        for i, (startX, startY, endX, endY) in enumerate(boxes):
            # draw the bounding box of the face along with the associated
            # probability
            if confidences[i] < min_confidence:
                continue
            text = "{:.2f}%".format(confidences[i] * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(res, (startX, startY), (endX, endY),
                          (0, 0, 255), 2)
            cv2.putText(res, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        return res


if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    default_model = "../models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
    ap.add_argument("-p", "--prototxt",
                    default="../models/deploy.prototxt",
                    help="path to Caffe 'deploy' prototxt file")
    ap.add_argument("-m", "--model",
                    default=default_model,
                    help="path to Caffe pre-trained model")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
                    help="minimum probability to filter weak detections")
    ap.add_argument("-i", "--image", type=str,
                    help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())

    detector = face_detector(args['prototxt'], args['model'])

    if args['image'] is None:
        cap = cv2.VideoCapture(0)

        while cv2.waitKey(1) != 27:
            ret, frame = cap.read()
            (boxes, confidences) = detector.detect(frame)
            cv2.imshow("Camera", detector.draw(frame, boxes, confidences,
                       min_confidence=0.1))
    else:
        frame = cv2.imread(args['image'])
        (boxes, confidences) = detector.detect(frame, 0.1)
        print('{} faces detected'.format(len(boxes)))
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Image", detector.draw(frame, boxes, confidences, 0.1))
        cv2.waitKey()
