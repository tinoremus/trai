from trai.computervision.capturevideo.capturevideo import FrameProcessor
from trai.components.tflitemodelobjectdetection import ObjectDetectionTfLiteModel
from trai.computervision.objectdetection.models.coraledgetpumodels import SSDMobileNetV1CocoQuantPostProcess
from trai.computervision.objectdetection.models.coraledgetpumodels import SSDMobileNetV2CocoQuantPostProcess
from trai.computervision.objectdetection.models.coraledgetpumodels import SSDMMobileNetV1FineTunedPet
from trai.computervision.capturevideo.capturevideo import VideoCapture
from dataclasses import dataclass
import numpy as np
import cv2
from PIL import Image


@dataclass()
class ObjectDetectionFrameCV2Processor(FrameProcessor):
    model: ObjectDetectionTfLiteModel or None

    def process_frame(self, frame: np.array) -> np.array:
        if self.model is None:
            return frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (self.model.input_image_width, self.model.input_image_height))
        input_data = np.expand_dims(frame_resized, axis=0)

        self.model.set_inputs([input_data])
        self.model.invoke()
        self.model.add_boxes_to_cv2_frame(frame=frame)
        return frame


# TEST ======================================================================================================
def capture_video_objectdetection():

    model = SSDMobileNetV2CocoQuantPostProcess()
    # model = SSDMobileNetV1CocoQuantPostProcess()
    # model = SSDMMobileNetV1FineTunedPet()
    # model.show(True)

    frame_processor = ObjectDetectionFrameCV2Processor(model=model)
    cap = VideoCapture(
        source=0,
        title=model.name,
        show_info=True,
        info_type='fps',
        info_location='lb',
        frame_processor=frame_processor,
    )
    # cap.show(True)
    cap.run()


def single_image_processing():
    model = SSDMobileNetV1CocoQuantPostProcess()
    # model.show(True)
    frame_processor = ObjectDetectionFrameCV2Processor(model=model)

    img = Image.open(r'/Users/tremus/Pictures/Pixel4/Camera/20151125_150657.jpg')
    # img.show('Test Image')
    frame = np.array(img)
    frame_processor.process_frame(frame)


if __name__ == '__main__':
    # single_image_processing()
    capture_video_objectdetection()




