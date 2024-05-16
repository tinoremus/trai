from enum import Enum
import os


class TfLiteModelSelector(Enum):

    def __new__(cls, *args, **kwargs):
        value = len(cls.__members__) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __init__(self, title: str, model_file_name: str, label_file_name: str):
        self.title = title
        self.model_file_name = model_file_name
        self.label_file_name = label_file_name

    @property
    def path(self) -> str:
        return os.path.dirname(os.path.abspath(__file__))

    def show(self, cmd_output: bool = True):
        print_string = '{:20}: {}'
        info = list()
        info.append(print_string.format('Title', self.title))
        info.append(print_string.format('Path', self.path))
        info.append(print_string.format('Model file name', self.model_file_name))
        info.append(print_string.format('Label file name', self.label_file_name))
        if cmd_output:
            for line in info:
                print(line)
        else:
            return info

