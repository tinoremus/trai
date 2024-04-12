import os
import cv2
import time
import numpy as np
from modeltestvideo.testtflitemodels.coraledgetpu.objectdetection.modelselection import GoogleCoralEdgeTpuModelSelector
from components.tflitemodel import ObjectDetectionTfLiteModel


def main(_model):
    if _model.interpreter is None:
        print('Model failed to initialize')
        return

    # define a video capture object
    cap = cv2.VideoCapture(0)
    image_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    image_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    frame_counter = 0
    start = time.time()
    title = 'Frame'
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

        # Object Detection Frame
        _model.add_boxes_to_cv2_frame(cv2, frame, image_width, image_height)

        if time.time() - start >= 1:
            print('Running at ca. {} Hz'.format(frame_counter))
            frame_counter = 0
            start = time.time()

        # Display the resulting frame
        cv2.imshow(title, frame)

        # the 'q' button to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object
    cap.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


# RUN =================================================================================================================
def run_object_detection():
    # SELECT MODEL
    select = GoogleCoralEdgeTpuModelSelector.SSD_MOBILENET_V2_COCO_QUANT_POSTPROCESS
    # select.show(True)

    model = ObjectDetectionTfLiteModel(
        name=select.name,
        link=os.path.join(select.path, select.model_file_name),
        label_link=os.path.join(select.path, select.label_file_name),
    )
    # model.show_labels(True)
    main(model)


if __name__ == '__main__':
    run_object_detection()
