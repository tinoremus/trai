import os
from dataclasses import dataclass
from tensorflow.lite.python.interpreter import Interpreter


@dataclass()
class TfLiteModel:
    name: str
    link: str

    @property
    def interpreter(self) -> Interpreter or None:
        if os.path.exists(self.link):
            interpreter = Interpreter(self.link)
            interpreter.allocate_tensors()
            return interpreter
        else:
            return None

    def invoke(self):
        self.interpreter.invoke() if self.interpreter is not None else None


@dataclass()
class ObjectDetectionTfLiteModel(TfLiteModel):
    label_link: str or None

    @property
    def labels(self) -> dict or None:
        encoding = 'utf-8'
        with open(self.label_link, 'r', encoding=encoding) as f:
            lines = f.readlines()
            if not lines:
                return {}
            if lines[0].split(' ', maxsplit=1)[0].isdigit():
                pairs = [line.split(' ', maxsplit=1) for line in lines]
                return {int(index): label.strip() for index, label in pairs}
            else:
                return {index: line.strip() for index, line in enumerate(lines)}

    @property
    def width(self) -> int:
        _width = -1
        if self.interpreter is not None:
            input_details = self.interpreter.get_input_details()
            # output_details = interpreter.get_output_details()
            _width = input_details[0]['shape'][2]
        return _width

    @property
    def height(self) -> int:
        _height = -1
        if self.interpreter is not None:
            input_details = self.interpreter.get_input_details()
            _height = input_details[0]['shape'][1]
        return _height

    def set_input(self, data):
        if self.interpreter is not None:
            input_details = self.interpreter.get_input_details()
            self.interpreter.set_tensor(input_details[0]['index'], data)

    def get_outputs(self) -> (list or None, list or None, list or None):
        if self.interpreter is not None:
            output_details = self.interpreter.get_output_details()
            _boxes = self.interpreter.get_tensor(output_details[0]['index'])[0]
            _classes = self.interpreter.get_tensor(output_details[1]['index'])[0]
            _scores = self.interpreter.get_tensor(output_details[2]['index'])[0]
            return _boxes, _classes, _scores
        else:
            return None, None, None

    def add_boxes_to_cv2_frame(self, cv2, frame, image_width, image_height):
        boxes, classes, scores = self.get_outputs()
        for i in range(len(scores)):
            if 0.5 < scores[i] <= 1.0:
                # Interpreter can return coordinates that are outside of image dimensions,
                # need to force them to be within image using max() and min()
                y_min = int(max(1, (boxes[i][0] * image_height)))
                x_min = int(max(1, (boxes[i][1] * image_width)))
                y_max = int(min(image_height, (boxes[i][2] * image_height)))
                x_max = int(min(image_width, (boxes[i][3] * image_width)))
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (10, 255, 0), 4)

                # Draw label
                if self.labels is not None:
                    object_name = self.labels[int(classes[i])]
                    label = '%s: %d%%' % (object_name, int(scores[i] * 100))

                    label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

                    # Make sure not to draw label too close to top of window
                    label_y_min = max(y_min, label_size[1] + 10)
                    cv2.rectangle(
                        frame,
                        (x_min, label_y_min - label_size[1] - 10),
                        (x_min + label_size[0], label_y_min + base_line - 10), (255, 255, 255), cv2.FILLED)
                    cv2.putText(
                        frame,
                        label,
                        (x_min, label_y_min - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    def show_labels(self, cmd_output: bool = True):
        info = list()
        info.append('LABELS from {}'.format(self.label_link))
        [info.append('  {:>3}: {}'.format(index, self.labels[index])) for index in self.labels]
        info.append('')

        if cmd_output:
            for line in info:
                print(line)
        else:
            return info
