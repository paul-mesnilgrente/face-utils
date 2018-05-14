# import the necessary packages
from facial_landmark_detection import FACIAL_LANDMARKS_IDXS
from facial_landmark_detection import shape_to_np
import facial_landmark_detection
import face_detection
import argparse
import numpy as np
import cv2

class face_aligner:
    def __init__(self, desiredLeftEye=(0.35, 0.35),
                 desiredFaceWidth=256, desiredFaceHeight=None):
        # store the facial landmark predictor, desired output left
        # eye position, and desired output face width + height
        self.desiredLeftEye = desiredLeftEye
        self.desiredFaceWidth = desiredFaceWidth
        self.desiredFaceHeight = desiredFaceHeight

        # if the desired face height is None, set it to be the
        # desired face width (normal behavior)
        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth

    def align(self, image, shapes, boxes):
        outputs = []
        for i, shape in enumerate(shapes):
            print(shape)
            rect = boxes[i]

            # extract the left and right eye (x, y)-coordinates
            (lStart, lEnd) = FACIAL_LANDMARKS_IDXS["left_eye"]
            (rStart, rEnd) = FACIAL_LANDMARKS_IDXS["right_eye"]
            leftEyePts = shape[lStart:lEnd]
            rightEyePts = shape[rStart:rEnd]

            # compute the center of mass for each eye
            leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
            rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

            # compute the angle between the eye centroids
            dY = rightEyeCenter[1] - leftEyeCenter[1]
            dX = rightEyeCenter[0] - leftEyeCenter[0]
            angle = np.degrees(np.arctan2(dY, dX)) - 180

            # compute the desired right eye x-coordinate based on the
            # desired x-coordinate of the left eye
            desiredRightEyeX = 1.0 - self.desiredLeftEye[0]

            # determine the scale of the new resulting image by taking
            # the ratio of the distance between eyes in the *current*
            # image to the ratio of distance between eyes in the
            # *desired* image
            dist = np.sqrt((dX ** 2) + (dY ** 2))
            desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
            desiredDist *= self.desiredFaceWidth
            scale = desiredDist / dist

            # compute center (x, y)-coordinates (i.e., the median point)
            # between the two eyes in the input image
            eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
                (leftEyeCenter[1] + rightEyeCenter[1]) // 2)

            # grab the rotation matrix for rotating and scaling the face
            M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

            # update the translation component of the matrix
            tX = self.desiredFaceWidth * 0.5
            tY = self.desiredFaceHeight * self.desiredLeftEye[1]
            M[0, 2] += (tX - eyesCenter[0])
            M[1, 2] += (tY - eyesCenter[1])

            # apply the affine transformation
            (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
            output = cv2.warpAffine(image, M, (w, h),
                flags=cv2.INTER_CUBIC)
            outputs.append(output)

        # return the aligned face
        return outputs

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

    # initialize the detectors
    detector = face_detection.face_detector(args["prototxt"], args["model"])
    predictor = facial_landmark_detection.facial_landmark_detector(args["shape_predictor"])
    aligner = face_aligner()

    cap = cv2.VideoCapture(0)

    while cv2.waitKey(1) != 27:
        ret, frame = cap.read()
        (boxes, confidences) = detector.detect(frame)
        shapes = predictor.predict(frame, boxes)
        outputs = aligner.align(frame, shapes, boxes)

        for i, image in enumerate(outputs):
            cv2.imshow("Face {}".format(i), image)

        frame = detector.draw(frame, boxes, confidences)
        frame = predictor.draw(frame, shapes)
        cv2.imshow("Camera", frame)