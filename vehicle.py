import logging
from donkeycar.vehicle import Vehicle

logger = logging.getLogger(__name__)

class MyVehicle(Vehicle):
    def __init__(self):
        super().__init__()
    
    ## Removed option for threading
    def add(self, part, inputs = [], outputs = [], run_condition = None):
        super().add(part, inputs, outputs, False, run_condition)
    
    

     