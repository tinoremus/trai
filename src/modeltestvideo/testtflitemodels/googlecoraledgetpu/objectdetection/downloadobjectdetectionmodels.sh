#!/bin/bash


download_from_test_data()
{
  BASE_DOMAIN="https://github.com/google-coral/edgetpu/raw/master/test_data/"
  FILENAME=$1
  if [ -e "$FILENAME" ]
  then
    echo "File $FILENAME already exists."
  else
    echo "downloading $FILENAME"
    wget $BASE_DOMAIN"$FILENAME"
  fi
}

download_from_test_data ssd_mobilenet_v1_coco_quant_postprocess.tflite
download_from_test_data ssd_mobilenet_v2_coco_quant_postprocess.tflite
download_from_test_data ssd_mobilenet_v1_fine_tuned_pet.tflite
download_from_test_data coco_labels.txt
download_from_test_data pet_labels.txt
download_from_test_data mobilenet_v1_1.0_224_ptq_float_io.tflite
download_from_test_data mobilenet_v2_1.0_224_quant.tflite
