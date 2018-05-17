#!/usr/bin/env python3

import requests
import argparse
import cv2


def draw_faces(image, json):
    image = cv2.imread(image)
    for i, face in enumerate(json['faces']):
        startX = face['up_left_point']['x']
        startY = face['up_left_point']['y']
        endX = face['down_right_point']['x']
        endY = face['down_right_point']['y']
        # draw the bounding box of the face along with the associated
        # probability
        cv2.rectangle(image, (startX, startY), (endX, endY),
                      (0, 0, 255), 3)

    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Image", image)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # Construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-u", "--user", required=True,
                    help="Your username")
    ap.add_argument("-p", "--password", required=True,
                    help="Your password")
    ap.add_argument("-i", "--image", type=str, required=True,
                    help="The image on which to call face detection")
    args = vars(ap.parse_args())

    # Create the authentication header
    headers = {
        'User-Token': args['user'],
        'Accept-Token': args['password']
    }

    # The form to submit
    files = {
        'filename': (args['image'], open(args['image'], 'rb'))
    }

    # Request the service to the platform
    url = 'https://demo.noos.cloud:9001/face_detection'
    response = requests.post(url, headers=headers, files=files)
    # Print the platform result
    json = response.json()
    print('Server response:', json)
    print('Type returned:', type(json))
    if json['error'] == '':
        print('{} face detected'.format(len(json['faces'])))
        draw_faces(args['image'], json)
        for i, face in enumerate(json['faces']):
            print('Face {}:'.format(i))
            print('    x_start:', face['up_left_point']['x'])
            print('    y_start:', face['up_left_point']['y'])
            print('    x_end:', face['down_right_point']['x'])
            print('    x_end:', face['down_right_point']['y'])
