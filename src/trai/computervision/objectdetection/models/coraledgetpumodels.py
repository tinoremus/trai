import os
from trai.components.tflitemodelobjectdetection import ObjectDetectionTfLiteModel
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))


class SSDMobileNetV1CocoQuantPostProcess(ObjectDetectionTfLiteModel):
    def __init__(self):
        # definition
        self.name = 'SSD_MOBILENET_V1_COCO_QUANT_POSTPROCESS'
        self.link = os.path.join(
            ROOT_PATH,
            'coraledgetpu',
            'ssd_mobilenet_v1_coco_quant_postprocess.tflite')
        self.label_file_path = os.path.join(
            ROOT_PATH,
            'coraledgetpu',
            'coco_labels.txt')

        # init
        self.load_labels()
        self.get_input_image_dim(pos=0)
        self.box_output = self.output_details[0]
        self.class_output = self.output_details[1]
        self.score_output = self.output_details[2]


class SSDMobileNetV2CocoQuantPostProcess(ObjectDetectionTfLiteModel):
    def __init__(self):
        # definition
        self.name = 'SSD_MOBILENET_V2_COCO_QUANT_POSTPROCESS'
        self.link = os.path.join(
            ROOT_PATH,
            'coraledgetpu',
            'ssd_mobilenet_v2_coco_quant_postprocess.tflite')
        self.label_file_path = os.path.join(
            ROOT_PATH,
            'coraledgetpu',
            'coco_labels.txt')

        # init
        self.load_labels()
        self.get_input_image_dim(pos=0)
        self.box_output = self.output_details[0]
        self.class_output = self.output_details[1]
        self.score_output = self.output_details[2]


class SSDMMobileNetV1FineTunedPet(ObjectDetectionTfLiteModel):
    def __init__(self):
        # definition
        self.name = 'SSD_MOBILENET_V1_FINE_TUNED_PET'
        self.link = os.path.join(
            ROOT_PATH,
            'coraledgetpu',
            'ssd_mobilenet_v1_fine_tuned_pet.tflite')
        self.label_file_path = os.path.join(
            ROOT_PATH,
            'coraledgetpu',
            'pet_labels.txt')

        # init
        self.load_labels()
        self.get_input_image_dim(pos=0)
        self.box_output = self.output_details[0]
        self.class_output = self.output_details[1]
        self.score_output = self.output_details[2]


# class MobileNetV110i224PtqFloatIo(ObjectDetectionTfLiteModel):
#     def __init__(self):
#         # definition
#         self.name = 'MOBILENET_V1_1_0_224_PTQ_FLOAT_IO'
#         self.link = os.path.join(
#             ROOT_PATH,
#             'coraledgetpu',
#             'mobilenet_v1_1.0_224_ptq_float_io.tflite')
#         self.label_file_path = os.path.join(
#             ROOT_PATH,
#             'coraledgetpu',
#             'coco_labels.txt')
#
#         # init
#         self.load_labels()
#         self.get_input_image_dim(pos=0)
#         self.box_output = self.output_details[0]
#         self.class_output = self.output_details[1]
#         self.score_output = self.output_details[2]


# class MobileNetV210i240Quant(ObjectDetectionTfLiteModel):
#     def __init__(self):
#         # definition
#         self.name = 'MOBILENET_V2_1_0_224_QUANT'
#         self.link = os.path.join(
#             ROOT_PATH,
#             'coraledgetpu',
#             'mobilenet_v2_1.0_224_quant.tflite')
#         self.label_file_path = os.path.join(
#             ROOT_PATH,
#             'coraledgetpu',
#             'coco_labels.txt')
#
#         # init
#         self.load_labels()
#         self.get_input_image_dim(pos=0)
#         self.box_output = self.output_details[0]
#         self.class_output = self.output_details[1]
#         self.score_output = self.output_details[2]


if __name__ == '__main__':
    m = SSDMobileNetV1CocoQuantPostProcess()
    m.show(True)



