import argparse
import cv2
import dlib


class face_detector:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()

    def detect(self, image):
        # image = imutils.resize(image, width=1000)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale frame
        return (image, self.detector(gray, 0))

    def draw(self, image, rects):
        for i, rect in enumerate(rects):
            # draw the bounding box of the face along with the associated
            # probability
            startX = rect.left()
            startY = rect.top()
            endX = rect.right()
            endY = rect.bottom()
            cv2.rectangle(image, (startX, startY), (endX, endY),
                          (0, 0, 255), 3)
        return image


def detection(window_title, detector, image):
    image, rects = detector.detect(image)
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.imshow(window_title, detector.draw(image, rects))


if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", type=str, required=True,
                    help="Image to run the face detection on")
    args = vars(ap.parse_args())

    detector = face_detector()

    # detect faces in the grayscale frame

    if args['image'] is None:
        cap = cv2.VideoCapture(0)

        while cv2.waitKey(1) != 27:
            ret, frame = cap.read()
            detection("Camera", detector, frame)
    else:
        detection("Image", detector, cv2.imread(args['image']))
        cv2.waitKey()
