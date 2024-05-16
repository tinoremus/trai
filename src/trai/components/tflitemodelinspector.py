from dataclasses import dataclass
from trai.components.tflitemodel import TfLiteModel
import os


@dataclass()
class TfLiteModelInspector:
    path: str
    file: str
    model: TfLiteModel or None = None

    def __post_init__(self):
        super().__init__()
        if not os.path.join(self.path, self.file):
            raise FileNotFoundError
        self.__load_model__()

    def __load_model__(self):
        link = os.path.join(self.path, self.file)
        self.model = TfLiteModel(name=self.file, link=link)

    def show(self, cmd_output: bool = True):
        print_string = '{}: {}'
        info = list()
        info.append(print_string.format('Path', self.path))
        info.append(print_string.format('File', self.file))
        info.append('')
        info += self.model.show(False)

        if cmd_output:
            for line in info:
                print(line)
        else:
            return info


if __name__ == '__main__':
    # path_name = r'/Volumes/Macintosh HD/Users/tremus/Documents/repos/github/trai/src/modeltestvideo/testtflitemodels/googlemediapipe/audioclassifier/'
    # file_name = 'yamnet_float32.tflite'
    path_name = r'/Users/tremus/Downloads/'
    file_name = 'autocomplete.tflite'
    mi = TfLiteModelInspector(path=path_name, file=file_name)
    mi.show(True)
