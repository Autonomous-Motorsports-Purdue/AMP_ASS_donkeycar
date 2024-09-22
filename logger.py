from donkeycar.parts.logger import LoggerPart

class Logger(LoggerPart):
    
    def add_inputs(self, inputs):
        for input in inputs:
            if input not in self.inputs:
                self.inputs.append(input)
                self.values[input] = None