import logging
from donkeycar.vehicle import Vehicle as BaseVehicle

logger = logging.getLogger(__name__)

class Vehicle(BaseVehicle):

    ## Removed option for threading
    def add(self, part, inputs = [], outputs = [], run_condition = None):
        super().add(part, inputs, outputs, False, run_condition)
    
    

     