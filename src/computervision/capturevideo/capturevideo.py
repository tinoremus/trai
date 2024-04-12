from dataclasses import dataclass
import datetime
import numpy
import cv2


@dataclass()
class FrameProcessor:
    def process_frame(self, frame) -> numpy.ndarray:
        return frame


@dataclass()
class VideoCapture:
    source: int
    exit_key: str = 'q'
    frame_counter: int = 0
    _cap: cv2.VideoCapture or None = None
    show_info: bool = False
    info_type: str = 'frame_counter'
    info_location: str = 'lt'
    info_text_font = cv2.FONT_HERSHEY_SIMPLEX
    frame_processor: FrameProcessor = FrameProcessor()

    def __post_init__(self):
        self.info_type = self.info_type \
            if self.info_type in [
                'frame_counter',
                'date_time',
        ] else 'frame_counter'
        self.info_location = (
            self.info_location) if self.info_location in [
            'lt', 'mt', 'rt',
            'lm', 'mm', 'rm',
            'lb', 'mb', 'rb',
        ] else 'lt'
        self._cap = cv2.VideoCapture(self.source)

    @property
    def capture_height(self) -> float:
        return self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    @property
    def capture_width(self) -> float:
        return self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    def __get_label_x_position__(self, label: str) -> int:
        if self.info_location.startswith('r'):
            label_size, _ = self.__get_label_dim__(label)
            dx = self.capture_width - label_size[0] - 20
        elif self.info_location.startswith('m'):
            label_size, _ = self.__get_label_dim__(label)
            dx = self.capture_width / 2 - label_size[1] / 2
        else:
            dx = 20
        return int(dx)

    def __get_label_y_position__(self, label: str) -> int:
        if self.info_location.endswith('b'):
            dy = self.capture_height - 20
        elif self.info_location.endswith('m'):
            label_size, _ = self.__get_label_dim__(label)
            dy = self.capture_height / 2 + label_size[1] / 2
        else:
            label_size, _ = self.__get_label_dim__(label)
            dy = 20 + label_size[1] / 2
        return int(dy)

    def __get_label_dim__(self, label: str) -> ((int, int), int):
        label_size, base_line = cv2.getTextSize(label, self.info_text_font, 0.7, 2)
        return label_size, base_line

    def __get_box_lb_coordinates__(self, label: str) -> (int, int):
        # (x_min, label_y_min - label_size[1] - 10)
        x = self.__get_label_x_position__(label) - 10
        y = self.__get_label_y_position__(label) + 10
        return x, y

    def __get_box_rt_coordinates__(self, label: str) -> (int, int):
        label_size, _ = self.__get_label_dim__(label)
        x = self.__get_label_x_position__(label) + label_size[0] + 10
        y = self.__get_label_y_position__(label) - label_size[1] - 10
        return x, y

    def run(self):
        if self._cap is None:
            return
        while True:
            self.frame_counter += 1
            # capture frame
            ret, frame = self._cap.read()
            # process frame
            frame = self.add_info(frame) if self.show_info else frame
            frame = self.frame_processor.process_frame(frame)
            # show frame
            cv2.imshow('frame', frame)
            # press "q" to end video capture
            if cv2.waitKey(1) & 0xFF == ord(self.exit_key):
                break
        # clean up
        self._cap.release()
        cv2.destroyAllWindows()

    def add_info(self, frame:numpy.ndarray) -> numpy.ndarray:
        if self.info_type == 'frame_counter':
            label = 'FRAME={}'.format(self.frame_counter)
        elif self.info_type == 'date_time':
            label = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S:%f')
        else:
            return frame

        cv2.rectangle(
            frame,
            self.__get_box_lb_coordinates__(label),
            self.__get_box_rt_coordinates__(label),
            (255, 255, 255),
            cv2.FILLED,
        )

        cv2.putText(
            frame, label,
            (self.__get_label_x_position__(label), self.__get_label_y_position__(label)),
            self.info_text_font,
            0.7,
            (0, 0, 0),
            2)
        return frame

    def show(self, cmd_output: bool = True, **kwargs):
        print_string = '{:30}: {}'
        info = list()
        info.append(print_string.format('Source ID', self.source))
        info.append(print_string.format('Capture width', self.capture_width))
        info.append(print_string.format('Capture height', self.capture_height))
        info.append(print_string.format('Press to stop', self.exit_key))

        if cmd_output:
            for line in info:
                print(line)
        else:
            return info


# TEST ======================================================================================================
def capture_video_fun():
    # select source by ID
    vid = cv2.VideoCapture(0)
    while True:
        # Capture the video frames
        ret, frame = vid.read()
        cv2.imshow('frame', frame)
        # press "q" to end video capture
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vid.release()
    cv2.destroyAllWindows()


def capture_video_obj():

    cap = VideoCapture(
        source=0,
        show_info=True,
        info_type='date_time',
        info_location='lb'
    )
    cap.show(True)
    cap.run()


if __name__ == '__main__':
    capture_video_obj()
