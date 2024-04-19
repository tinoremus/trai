import os
from dataclasses import dataclass, field
from typing import List
from tensorflow.lite.python.interpreter import Interpreter
import numpy
import time


@dataclass()
class TfLiteModelInOutputDetails:
    pos: int
    location: str
    name: str
    index: int
    shape: numpy.array
    shape_signature: any
    dtype: str
    quantization: tuple
    quantization_parameters: dict
    sparsity_parameters: dict

    @classmethod
    def from_dict(cls, pos: int, location: str,details: dict):
        return cls(
            pos=pos,
            location=location,
            name=details['name'],
            index=details['index'],
            shape=details['shape'],
            shape_signature=details['shape_signature'],
            dtype=details['dtype'],
            quantization=details['quantization'],
            quantization_parameters=details['quantization_parameters'],
            sparsity_parameters=details['sparsity_parameters'],
        )

    def show(self, cmd_output: bool = False):
        print_string = '{:25}: {}'
        info = list()
        info.append(print_string.format('Pos', self.pos))
        info.append(print_string.format('Location', self.location))
        info.append(print_string.format('Index', self.index))
        info.append(print_string.format('Name', self.name))
        info.append(print_string.format('Shape', self.shape))
        info.append(print_string.format('Shape Signature', self.shape_signature))
        info.append(print_string.format('dtype', self.dtype))
        info.append(print_string.format('Quantization', self.quantization))
        info.append(print_string.format('Quantization Parameters', self.quantization_parameters))
        info.append(print_string.format('Sparsity Parameters', self.sparsity_parameters))
        if cmd_output:
            for line in info:
                print(line)
        else:
            return info


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

    def set_inputs(self, data):
        if self.interpreter is not None:
            for ind in self.input_details:
                self.interpreter.set_tensor(ind.index, data)

    def invoke(self):
        self.interpreter.invoke() if self.interpreter is not None else None

    def get_outputs(self) -> list:
        data = list()
        if self.interpreter is not None:
            for ond in self.output_details:
                out = self.interpreter.get_tensor(ond.index)[0]
                data.append(out)
        return data

    @property
    def expected_inference_time(self) -> float:
        return self.dummy_run()

    def dummy_run(self) -> float:
        self.set_inputs(self.dummy_input_data)
        samples = []
        for loop in range(10):
            start = time.time()
            self.invoke()
            samples.append(time.time() - start)
        return numpy.average(samples) if samples else 0.0

    @property
    def input_details(self) -> List[TfLiteModelInOutputDetails]:
        details = list()
        if self.interpreter is not None:
            for ind, entry in enumerate(self.interpreter.get_input_details()):
                ip = TfLiteModelInOutputDetails.from_dict(pos=ind, location='Input', details=entry)
                details.append(ip)
        return details

    @property
    def dummy_input_data(self) -> list:
        input_data = list()
        for ind in self.input_details:
            size = ind.shape[1:]
            if numpy.issubdtype(ind.dtype, numpy.integer):
                dummy = numpy.random.random_integers(255, size=size)
            else:
                dummy = numpy.random.random_sample(size)
            input_data.append(dummy.astype(ind.dtype.__name__))
        return input_data

    @property
    def output_details(self) -> List[TfLiteModelInOutputDetails]:
        details = list()
        if self.interpreter is not None:
            for ond, entry in enumerate(self.interpreter.get_output_details()):
                op = TfLiteModelInOutputDetails.from_dict(pos=ond, location='Output', details=entry)
                details.append(op)
        return details

    def get_dummy_inputs(self) -> list:
        input_data = list()
        for ip in self.input_details:
            dummy = []
            input_data.append(dummy)

        return input_data

    def show_input_details(self, cmd_output: bool = True):
        info = list()
        info.append('INPUT DETAILS:')
        if self.interpreter is None:
            info.append('  Interpreter not available.')
        else:
            for ip in self.input_details:
                info.append(f'  Input {ip.pos}:')
                info += [f'    {row}' for row in ip.show(False)]
                info.append('')

        if cmd_output:
            for line in info:
                print(line)
        else:
            return info

    def show_dummy_input_data(self, cmd_output: bool = True):
        print_string = '    {:25}: {}'
        info = list()
        info.append('DUMMY INPUT DATA:')
        for i, di in enumerate(self.dummy_input_data):
            info.append(f'  Input {i}:')
            info.append(print_string.format('Shape', di.shape))
            info.append(print_string.format('dType', di.dtype))
        info.append('')

        if cmd_output:
            for line in info:
                print(line)
        else:
            return info

    def show_output_details(self, cmd_output: bool = True):
        info = list()

        info.append('OUTPUT DETAILS:')
        if self.interpreter is None:
            info.append('  Interpreter not available.')
        else:
            for op in self.output_details:
                info.append(f'  Output {op.pos}:')
                info += [f'    {row}' for row in op.show(False)]
                info.append('')

        if cmd_output:
            for line in info:
                print(line)
        else:
            return info

    def show_outputs(self, cmd_output: bool = True):
        print_string = '    {:25}: {}'
        info = list()
        info.append('OUTPUT DATA:')
        for i, do in enumerate(self.get_outputs()):
            info.append(f'  Output {i}:')
            info.append(print_string.format('Shape', do.shape))
            info.append(print_string.format('dType', do.dtype))
        info.append('')

        if cmd_output:
            for line in info:
                print(line)
        else:
            return info

    def show(self, cmd_output: bool = True):
        print_string = '{:29}: {}'
        info = list()
        info.append(print_string.format('Name', self.name))
        info.append(print_string.format('Link', self.link))
        info.append(print_string.format('Expected Inference Time', self.expected_inference_time))
        info.append('')
        info += self.show_input_details(False)
        info += self.show_dummy_input_data(False)
        info.append('')
        info += self.show_output_details(False)
        info += self.show_outputs(False)
        info.append('')

        if cmd_output:
            for line in info:
                print(line)
        else:
            return info
