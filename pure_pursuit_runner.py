import donkeycar as dk
from donkeycar.parts.controller import LocalWebController

import time
from parts.health_check import HealthCheck
from parts.uart import UART_Driver
from parts.uart_backup import UART_backup_driver
from constants import DRIVE_LOOP_HZ
from parts.logger import Logger
from parts.frame_publisher import Frame_Publisher
from parts.segment_model import Segment_Model
from parts.pure_pursuit import Pure_Pursuit

if __name__ == "__main__":
    # web controller
    V = dk.vehicle.Vehicle()
    heartbeat= HealthCheck("192.168.1.100", 6000) # aryamaan has ip 100
    V.add(heartbeat, inputs=[], outputs=["safety/heartbeat"])
    V.add(Pure_Pursuit(1.000506), inputs=["waypoint"], outputs=["controls/steering","controls/throttle"])
    #V.add(Frame_Publisher(), outputs=['sensors/ZED/RGB/left', 'sensors/ZED/RGB/right'], threaded=False)
    #V.add(Segment_Model(), inputs=['sensors/ZED/RGB/left'], outputs=['perception/segmentedTrack', 'centroid','controls/steering', 'controls/throttle'])
    #V.add(LaneDetect(), inputs=['sensors/ZED/RGB/left', ' ', ' '], outputs=['points', 'perception/segmentedTrack'], threaded=False)
    #V.add(Logger(), inputs=['sensors/ZED/RGB/left', 'perception/segmentedTrack', 'centroid', 'controls/steering', 'controls/throttle'], threaded=False)
    
    #controller = LocalWebController()
    # web controller just expects all these things even though they don't exist
    """
    V.add(
        controller,
        inputs=["perception/segmentedTrack", "tub/num_records", "user/mode", "recording"],
        outputs=[
            "user/steering_fake",
            "user/throttle_fake",
            "user/mode",
            "recording",
            "web/buttons",
        ],
        threaded=True,
    )
    """

    # uart controller
    uart = UART_backup_driver("/dev/ttyACM0")
    V.add(
        uart,
        inputs=["controls/throttle", "controls/steering", "safety/heartbeat"],
        outputs=[],
        threaded=False,
    )

    V.start(rate_hz=DRIVE_LOOP_HZ)
