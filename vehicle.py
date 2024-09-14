import time
import numpy as np
import logging
from threading import Thread

logger = logging.getLogger(__name__)

class Vehicle:

    """
    Initialize vehicle with memory, parts, threads and on status
    """
    def __init__(self, memory = None):
        if not memory:
            memory = {}
        self.memory = memory
        self.parts = []
        self.on = True
        self.threads = []
    

    def add(self, part, inputs = [], outputs = [], threaded = False, run_condition = None):
        """
        Function to add parts to the vehicle

        Parameters
            part: class
                donkey car part with run() attribute
            inputs: list
                channel names to retrieve from memory
            outputs: list
                channel names to save to memory
            threaded: boolean
                if part should be run in a seperate thread
            run_condition: str
                if part should be run
        """

        p = part
        ## Logging information for part
        logger.info('Adding part {}.'.format(p.__class__.__name__))
        entry = {}
        entry['part'] = p
        entry['inputs'] = inputs
        entry['outputs'] = outputs
        entry['run_condition'] = run_condition

        if threaded:
            t = Thread(target = part.update, args = ())
            t.daemon = True
            entry['thread'] = t
        
        self.parts.append(entry)

    def remove(self, part):
        ## Remove part from list
        self.parts.remove(part)
    

     