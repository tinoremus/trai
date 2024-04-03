"""
https://gist.github.com/Namburger/f7e6c18af94ef3d6a70076a130eb1f7c
https://github.com/google-coral/edgetpu/tree/master/test_data
"""
import os
import cv2
import time
import numpy as np
from components.tflitemodel import ObjectDetectionTfLiteModel
from enum import Enum


class ObjectDetectionModel(Enum):

    def __new__(cls, *args, **kwargs):
        value = len(cls.__members__) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __init__(self, title: str, link: str, label_link: str):
        self.title = title
        self.link = link
        self.label_link = label_link

    SSD_MOBILENET_V1_COCO_QUANT_POSTPROCESS = \
        'SSD_MOBILENET_V1_COCO_QUANT_POSTPROCESS', \
        os.path.join(os.getcwd(), 'ssd_mobilenet_v1_coco_quant_postprocess.tflite'), \
        os.path.join(os.getcwd(), 'coco_labels.txt')
    SSD_MOBILENET_V1_FINE_TUNED_PET = \
        'SSD_MOBILENET_V1_COCO_QUANT_POSTPROCESS', \
        os.path.join(os.getcwd(), 'ssd_mobilenet_v1_fine_tuned_pet.tflite'), \
        os.path.join(os.getcwd(), 'pet_labels.txt')
    SSD_MOBILENET_V2_COCO_QUANT_POSTPROCESS = \
        'SSD_MOBILENET_V2_COCO_QUANT_POSTPROCESS', \
        os.path.join(os.getcwd(), 'ssd_mobilenet_v2_coco_quant_postprocess.tflite'), \
        os.path.join(os.getcwd(), 'coco_labels.txt')
    MOBILENET_V1_1_0_224_PTQ_FLOAT_IO = \
        'MOBILENET_V1_1_0_224_PTQ_FLOAT_IO', \
            os.path.join(os.getcwd(), 'mobilenet_v1_1.0_224_ptq_float_io.tflite'), \
            os.path.join(os.getcwd(), 'coco_labels.txt')
    MOBILENET_V2_1_0_224_QUANT = \
        'MOBILENET_V1_1_0_224_PTQ_FLOAT_IO', \
            os.path.join(os.getcwd(), 'mobilenet_v2_1.0_224_quant.tflite'), \
            os.path.join(os.getcwd(), 'coco_labels.txt')


def main(_model):
    if model.interpreter is None:
        print('Model failed to initialize')
        return

    # define a video capture object
    cap = cv2.VideoCapture(0)
    image_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    image_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    frame_counter = 0
    start = time.time()
    while True:
        frame_counter += 1
        # Capture the video frame by frame
        ret, frame = cap.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (_model.width, _model.height))
        input_data = np.expand_dims(frame_resized, axis=0)

        # set frame as input tensors
        _model.set_input(input_data)
        _model.invoke()
        _model.add_boxes_to_cv2_frame(cv2, frame, image_width, image_height)

        if time.time() - start >= 1:
            print('fps:', frame_counter)
            frame_counter = 0
            start = time.time()

        # Display the resulting frame
        cv2.imshow('frame', frame)

        # the 'q' button to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object
    cap.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # select = ObjectDetectionModel.SSD_MOBILENET_V1_COCO_QUANT_POSTPROCESS
    select = ObjectDetectionModel.SSD_MOBILENET_V2_COCO_QUANT_POSTPROCESS
    print('Selected Model: {}'.format(select.name))
    model = ObjectDetectionTfLiteModel(name=select.name, link=select.link, label_link=select.label_link)
    main(model)
