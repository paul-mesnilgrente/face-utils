# face-utils

## I. Installation

###  I.1 Python

```bash
pyenv virtualenv face-utils
pip install -U pip
pip install numpy
pip install opencv-python
pip install dlib
```

### I.2 C++

**Prerequisites:**

- OpenCV
- DLib

```bash
mkdir build && cd build
cmake ..
make
```

### I.3 Download the models

```bash
mkdir models && cd models
github=https://raw.githubusercontent.com
wget ${github}/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel
wget ${github}/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
wget https://github.com/AKSHAYUBHAT/TensorFace/raw/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat
```
