import os
from dataclasses import dataclass, field
from typing import List
from trai.components.tflitemodel import TfLiteModel, TfLiteModelInOutputDetails
import cv2
import numpy as np


@dataclass()
class ObjectBox:
    img_width: int
    img_height: int
    box: List[int]
    object_name: str or None = None
    score: float or None = None

    @property
    def x_min(self) -> int:
        return int(max(1, (self.box[1] * self.img_width)))

    @property
    def x_max(self) -> int:
        return int(min(self.img_width, (self.box[3] * self.img_width)))

    @property
    def y_min(self) -> int:
        return int(max(1, (self.box[0] * self.img_height)))

    @property
    def y_max(self) -> int:
        return int(min(self.img_height, (self.box[2] * self.img_height)))

    @property
    def label(self) -> str:
        return '{}: {:00.2f}%'.format(self.object_name, self.score * 100)

    def get_label_size_and_base_line(self) -> (int, (int, int)):
        label_size, base_line = cv2.getTextSize(self.label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        return label_size, base_line

    @property
    def base_line(self) -> int:
        _, bl = self.get_label_size_and_base_line()
        return bl

    @property
    def label_width(self) -> int:
        ls, _ = self.get_label_size_and_base_line()
        return ls[0]

    @property
    def label_height(self) -> int:
        ls, _ = self.get_label_size_and_base_line()
        return ls[1]

    @property
    def label_x_min(self) -> int:
        return self.x_min

    @property
    def label_y_min(self) -> int:
        return max(self.y_min, self.label_height + 10) - self.label_height

    @property
    def label_x_max(self) -> int:
        return 0

    @property
    def label_y_max(self) -> int:
        return 0


@dataclass()
class ObjectDetectionTfLiteModel(TfLiteModel):
    label_file_path: str or None = None
    labels: dict = field(default_factory=dict)
    label_count: int = 0
    input_image_width: int = 0
    input_image_height: int = 0

    box_output: TfLiteModelInOutputDetails or None = None
    class_output: TfLiteModelInOutputDetails or None = None
    score_output: TfLiteModelInOutputDetails or None = None

    boxes: list = field(default_factory=list)
    classes: list = field(default_factory=list)
    scores: list = field(default_factory=list)

    def load_labels(self):
        if not os.path.exists(self.label_file_path):
            self.labels = {}
            self.label_count = 0
            return
        encoding = 'utf-8'
        with open(self.label_file_path, 'r', encoding=encoding) as f:
            lines = f.readlines()
            if not lines:
                self.labels = {}
                self.label_count = 0
            if lines[0].split(' ', maxsplit=1)[0].isdigit():
                pairs = [line.split(' ', maxsplit=1) for line in lines]
                self.labels = {int(index): label.strip() for index, label in pairs}
                self.label_count = len(self.labels)
            else:
                self.labels = {index: line.strip() for index, line in enumerate(lines)}
                self.label_count = len(self.labels)

    def show_labels(self, cmd_output: bool = True):
        info = list()
        info.append('LABELS from {}'.format(self.label_file_path))
        [info.append('  {:>3}: {}'.format(index, self.labels[index])) for index in self.labels]
        info.append('')

        if cmd_output:
            for line in info:
                print(line)
        else:
            return info

    def get_input_image_dim(self, pos: int):
        inputs_info = [inf for inf in self.input_details if inf.pos == pos]
        if inputs_info:
            input_info = inputs_info[0]
            self.input_image_height = input_info.shape[1]
            self.input_image_width = input_info.shape[2]

    def fetch_outputs(self):
        if self.interpreter is not None:
            if self.box_output is not None:
                self.boxes = self.interpreter.get_tensor(self.box_output.index)[0]
            if self.class_output is not None:
                self.classes = self.interpreter.get_tensor(self.class_output.index)[0]
                self.classes = [int(c) if not np.isnan(c) else -1 for c in self.classes]
            if self.score_output is not None:
                self.scores = self.interpreter.get_tensor(self.score_output.index)[0]

    def get_boxes(self, frame: np.array) -> List[ObjectBox]:
        self.fetch_outputs()
        ret = [(s, b, c) for s, b, c in zip(self.scores, self.boxes, self.classes) if 1 >= s > 0.5]
        return [
            ObjectBox(
                img_width=frame.shape[1],
                img_height=frame.shape[0],
                box=result[1],
                object_name=self.__get_label__(result),
                score=result[0],
            )
            for result in ret] if ret else []

    def __get_label__(self, result) -> str or None:
        label = None
        try:
            if self.labels is not None:
                if self.label_count >= result[2]:
                    label = self.labels[result[2]]
        except Exception as err:
            print(err)
        return label

    def add_boxes_to_frame(self, frame: np.array):

        [self.__cv2_boxes__(frame, box) for box in self.get_boxes(frame)]

    @staticmethod
    def __cv2_boxes__(frame: np.array, box: ObjectBox):
        object_frame_color = (10, 255, 0)
        label_frame_fill_color = (255, 255, 255)
        label_text_color = (0, 0, 0)

        cv2.rectangle(frame, (box.x_min, box.y_min), (box.x_max, box.y_max), object_frame_color, 4)
        if box.object_name is not None:
            cv2.rectangle(frame,
                          (box.x_min, box.label_y_min + 10),
                          (box.x_min + box.label_width, box.label_y_min + box.base_line - 30),
                          label_frame_fill_color, cv2.FILLED)
            cv2.putText(frame, box.label, (box.x_min, box.label_y_min),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_text_color, 2)

    @property
    def tflite_model_object_detection_info(self) -> list:
        print_string = '  {:27}: {}'
        info = self.tflite_model_info
        info.append('')
        info.append('LABELS:')
        info.append(print_string.format('Label File Path', self.label_file_path))
        info.append(print_string.format('Label Count', self.label_count))
        info.append(print_string.format('Input Image Width', self.input_image_width))
        info.append(print_string.format('Input Image Height', self.input_image_height))
        info.append('')
        info.append(print_string.format('Box Output Pos', self.box_output))
        info.append(print_string.format('Class Output Pos', self.class_output))
        info.append(print_string.format('Score Output Pos', self.score_output))
        info.append('')
        return info

    def show(self, cmd_output: bool = True):

        info = self.tflite_model_object_detection_info
        if cmd_output:
            for line in info:
                print(line)
        else:
            return info
