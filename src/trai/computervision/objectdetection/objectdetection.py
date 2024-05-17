import PIL.Image

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
    confidence_threshold: float = 0.7

    def process_frame(self, frame: np.array) -> np.array:
        if self.model is None:
            return frame
        self.model.confidence_threshold = self.confidence_threshold
        # switch order from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # resize image to fit model inputs
        frame_resized = cv2.resize(frame_rgb, (self.model.input_image_width, self.model.input_image_height))
        # add batch dimension
        input_data = np.expand_dims(frame_resized, axis=0)

        # run inference
        self.model.set_inputs([input_data])
        self.model.invoke()

        # add boxes and annotation to frame
        self.model.add_boxes_to_frame(frame=frame)
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
    model = SSDMobileNetV2CocoQuantPostProcess()
    # model.show(True)
    frame_processor = ObjectDetectionFrameCV2Processor(model=model)

    raw_img = Image.open(r'/Users/tremus/Pictures/Pixel4/Camera/20151125_150657.jpg')
    # raw_img.show('Raw Image')

    raw_frame = np.array(raw_img)
    processed_frame = frame_processor.process_frame(raw_frame)

    processed_image = PIL.Image.fromarray(processed_frame)
    # processed_image = Image.fromarray(np.uint8(processed_frame * 255))  # cool effect
    processed_image.show('Processed Image')


if __name__ == '__main__':
    single_image_processing()
    # capture_video_objectdetection()




