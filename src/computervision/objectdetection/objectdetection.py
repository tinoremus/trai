from computervision.capturevideo.capturevideo import VideoCapture, FrameProcessor
from computervision.objectdetection.models.tfliteobjectdetectionmodel import ObjectDetectionTfLiteModel
from computervision.objectdetection.models.coraledgetpu.modelselection import GoogleCoralEdgeTpuModelSelector
from dataclasses import dataclass
import numpy
import cv2
import os


@dataclass()
class ObjectDetectionFrameProcessor(FrameProcessor):
    model: ObjectDetectionTfLiteModel or None

    def process_frame(self, frame) -> numpy.ndarray:
        self.model.add_boxes_to_cv2_frame(frame=frame, image_width=100, image_height=100)
        return frame


# TEST ======================================================================================================
def capture_video_objectdetection():

    select = GoogleCoralEdgeTpuModelSelector.SSD_MOBILENET_V1_COCO_QUANT_POSTPROCESS
    # select.show()
    
    model = ObjectDetectionTfLiteModel(
        name=select.name,
        link=os.path.join(select.path, select.model_file_name),
        label_link=os.path.join(select.path, select.label_file_name),
    )
    model.show(True)
    # frame_processor = ObjectDetectionFrameProcessor(model=model)
    #
    # cap = VideoCapture(
    #     source=0,
    #     show_info=True,
    #     info_type='date_time',
    #     info_location='lb',
    #     frame_processor=frame_processor,
    # )
    # cap.show(True)
    # cap.run()


if __name__ == '__main__':
    capture_video_objectdetection()




