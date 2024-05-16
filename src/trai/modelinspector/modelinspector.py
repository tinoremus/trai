from trai.components.tflitemodelinspector import TfLiteModelInspector
import flet as ft
import os
import sys
TEXT_SIZE = 14


class ModelInspector(ft.UserControl):
    def __init__(self, page):
        super().__init__()

        if getattr(sys, 'frozen', False):
            self.home_path = os.path.dirname(sys.executable)
        elif __file__:
            self.home_path = os.path.dirname(__file__)

        page.snack_bar = ft.SnackBar(
            content=ft.Text("Snackbar"),
            action="Alright!",
        )

        self.le_file_path: ft.TextField = ft.TextField(
            label='Model Filepath',
            text_size=TEXT_SIZE,
            text_align=ft.TextAlign.LEFT,
            hint_text="TFLite or ONNX File",
            expand=True,
            read_only=False
        )
        self.pick_files_dialog = ft.FilePicker(on_result=self.pick_files_result)
        page.overlay.append(self.pick_files_dialog)
        self.last_selected_path = self.home_path

        self.te_results = ft.Text(
            font_family="Courier",
            selectable=True,
            no_wrap=True,
            value="Results"
        )
        cr = ft.Row(
            controls=[self.te_results],
            scroll=ft.ScrollMode.ALWAYS,
            vertical_alignment=ft.CrossAxisAlignment.START,
        )
        self.result_column = ft.Column(
            spacing=10,
            controls=[cr],
            scroll=ft.ScrollMode.ALWAYS,
        )
        page.on_resize = self.page_resize

    def build(self):
        # FILE SELECTION
        file_selector = ft.Row(
                controls=[
                    self.le_file_path,
                    ft.FloatingActionButton(icon=ft.icons.FILE_OPEN, on_click=self.pick_file),
                ],
            )

        # FUNCTION BUTTONS
        clear_button = ft.OutlinedButton(text="Clear", on_click=self.clear_results)
        inspect_button = ft.OutlinedButton(text="Inspect File", on_click=self.inspect)
        inputs_button = ft.OutlinedButton(text="Dummy Inputs", on_click=self.show_inputs)
        run_button = ft.OutlinedButton(text="Run Model", on_click=self.run_model)
        outputs_button = ft.OutlinedButton(text="Dummy Outputs", on_click=self.show_outputs)
        inputs_clipboard_button = ft.OutlinedButton(text="Copy Inputs", on_click=self.inputs_to_clipboard)
        function_buttons = ft.Row(
            controls=[
                clear_button,
                inspect_button,
                inputs_button,
                run_button,
                outputs_button,
                inputs_clipboard_button,
            ]
        )

        # RESULTS
        self.result_column.height = self.page.height - 200
        self.result_column.width = self.page.width - 20
        results = ft.Container(
            self.result_column,
            margin=10,
            # border=ft.border.all(1),
        )

        # COMBINED LAYOUT
        main_layout = ft.Column(
            controls=[
                file_selector,
                function_buttons,
                ft.VerticalDivider(),
                results,
            ]
        )

        return main_layout

    def page_resize(self, e):
        self.result_column.height = self.page.window_height - 200
        self.result_column.width = self.page.window_width - 20
        self.update()

    def pick_file(self, event):
        self.pick_files_dialog.pick_files(
            dialog_title='Select Model File',
            initial_directory=self.last_selected_path,
            allowed_extensions=['tflite', 'onnx'],
        )

    def pick_files_result(self, event):
        file_info = event.files[0] if event.files else None
        if file_info is None:
            return
        self.clear_results(None)
        self.last_selected_path = os.path.dirname(file_info.path)
        self.le_file_path.value = file_info.path
        self.le_file_path.update()

    def show_snack_bar_message(self, event, msg: str):
        self.page.snack_bar = ft.SnackBar(ft.Text(msg))
        self.page.snack_bar.open = True
        self.page.update()

    def clear_results(self, event):
        self.te_results.value = ''
        self.update()

    def __get_model__(self) -> TfLiteModelInspector:
        file_name = self.le_file_path.value
        try:
            tmi = TfLiteModelInspector(
                path=os.path.dirname(file_name),
                file=os.path.basename(file_name)

            )
        except Exception as err:
            tmi = None
        return tmi

    def inspect(self, event):
        self.clear_results(None)
        tmi = self.__get_model__()
        self.te_results.value = '\n'.join(tmi.show(False))
        self.update()

    def show_inputs(self, event):
        self.clear_results(None)
        tmi = self.__get_model__()
        info = list()
        for data in tmi.model.dummy_input_data:
            info.append(f'Shape: {data.shape}')
            info.append(f'Type: {data.dtype}')
            info.append(repr(data))
            info.append('')
        self.te_results.value = '\n'.join(info)
        self.update()

    def run_model(self, event):
        self.clear_results(None)
        tmi = self.__get_model__()
        if tmi is None:
            return
        values = list()
        for vid, val in enumerate(tmi.model.dummy_run(loops=20), start=1):
            values.append(val)
            self.te_results.value += 'run {:>3} took {} sec\n'.format(vid, val)
            self.update()

        avg_time = sum(values) / len(values)
        avg_frq = 1 / avg_time
        info = ['\n', 'Average inference time = {} sec or {:.3f} Hz'.format(avg_time, avg_frq)]
        self.te_results.value += '\n'.join(info)
        self.update()

    def show_outputs(self, event):
        self.clear_results(None)
        tmi = self.__get_model__()
        info = list()
        for data in tmi.model.dummy_output_data:
            info.append(f'Shape: {data.shape}')
            info.append(f'Type: {data.dtype}')
            info.append(repr(data))
            info.append('')
        self.te_results.value = '\n'.join(info)
        self.update()

    def inputs_to_clipboard(self, event):
        tmi = self.__get_model__()
        text = str([item.tolist() for item in tmi.model.dummy_input_data]) \
            if tmi is not None else 'No Model Loaded'
        self.page.set_clipboard(text)
        self.show_snack_bar_message(None, msg="Copied Input Data to Clipboard")


def main(page: ft.Page):
    page.fonts = {
        "Courier": "fonts/Courier.ttc",
    }

    page.title = "Model Inspector App"
    page.theme = ft.theme.Theme(color_scheme_seed="green")
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.window_width = 1000
    page.update()

    # create application instance
    mfs = ModelInspector(page)
    # add application's root control to the page
    page.add(mfs)


if __name__ == '__main__':
    ft.app(main)
