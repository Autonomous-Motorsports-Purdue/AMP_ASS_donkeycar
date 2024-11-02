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

if __name__ == "__main__":
    # web controller
    V = dk.vehicle.Vehicle()
    health_check = HealthCheck("192.168.1.100", 6000) # aryamaan has ip 100
    V.add(health_check, inputs=[], outputs=["critical/health_check"])
    V.add(Frame_Publisher(), outputs=['left', 'right'], threaded=False)
    V.add(Segment_Model(), inputs=['left'], outputs=['overlay', 'centroid','user/steering', 'user/throttle'])
    #V.add(LaneDetect(), inputs=['left', ' ', ' '], outputs=['points', 'overlay'], threaded=False)
    V.add(Logger(), inputs=['left', 'overlay', 'centroid', 'user/steering', 'user/throttle'], threaded=False)
    
    controller = LocalWebController()
    # web controller just expects all these things even though they don't exist
    V.add(
        controller,
        inputs=["overlay", "tub/num_records", "user/mode", "recording"],
        outputs=[
            "user/steering_fake",
            "user/throttle_fake",
            "user/mode",
            "recording",
            "web/buttons",
        ],
        threaded=True,
    )

    # uart controller
    uart = UART_backup_driver("/dev/ttyACM0")
    V.add(
        uart,
        inputs=["user/throttle", "user/steering", "critical/health_check"],
        outputs=[],
        threaded=False,
    )

    V.start(rate_hz=DRIVE_LOOP_HZ)
