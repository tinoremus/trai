"""
https://gist.github.com/Namburger/f7e6c18af94ef3d6a70076a130eb1f7c
https://github.com/google-coral/edgetpu/tree/master/test_data
"""
import os
from trai.components import TfLiteModelSelector


class GoogleCoralEdgeTpuModelSelector(TfLiteModelSelector):

    @property
    def path(self) -> str:
        return os.path.dirname(os.path.abspath(__file__))

    SSD_MOBILENET_V1_COCO_QUANT_POSTPROCESS = \
        'SSD_MOBILENET_V1_COCO_QUANT_POSTPROCESS', \
        'ssd_mobilenet_v1_coco_quant_postprocess.tflite', \
        'coco_labels.txt'
    SSD_MOBILENET_V1_FINE_TUNED_PET = \
        'SSD_MOBILENET_V1_FINE_TUNED_PET', \
        'ssd_mobilenet_v1_fine_tuned_pet.tflite', \
        'pet_labels.txt'
    SSD_MOBILENET_V2_COCO_QUANT_POSTPROCESS = \
        'SSD_MOBILENET_V2_COCO_QUANT_POSTPROCESS', \
        'ssd_mobilenet_v2_coco_quant_postprocess.tflite', \
        'coco_labels.txt'
    # MOBILENET_V1_1_0_224_PTQ_FLOAT_IO = \
    #     'MOBILENET_V1_1_0_224_PTQ_FLOAT_IO', \
    #     'mobilenet_v1_1.0_224_ptq_float_io.tflite', \
    #     'coco_labels.txt'
    # MOBILENET_V2_1_0_224_QUANT = \
    #     'MOBILENET_V2_1_0_224_QUANT', \
    #     'mobilenet_v2_1.0_224_quant.tflite', \
    #     'coco_labels.txt'
