import donkeycar as dk
from donkeycar.parts.controller import LocalWebController

import time
from parts.health_check import HealthCheck
from parts.uart import UART_Driver
from parts.uart_backup import UART_backup_driver
from constants import DRIVE_LOOP_HZ
<<<<<<< HEAD
from parts.logger import Logger
from parts.frame_publisher import Frame_Publisher
from parts.lane_detect import LaneDetect

=======
from parts.uart_backup import UART_backup_driver
>>>>>>> 716b13a5bd616ae0c56dac041a8fd976b53cd3ca

if __name__ == "__main__":
    # web controller
    V = dk.vehicle.Vehicle()
    health_check = HealthCheck("192.168.1.100", 6000)
    V.add(health_check, inputs=[], outputs=["critical/health_check"])
    V.add(Frame_Publisher(), outputs=['left', 'right'], threaded=False)
    V.add(LaneDetect(), inputs=['left', ' ', ' '], outputs=['points', 'overlay'], threaded=False)
    V.add(Logger(), inputs=['left', 'right', 'points'], threaded=False)
    
    controller = LocalWebController()
    # web controller just expects all these things even though they don't exist
    V.add(
        controller,
        inputs=["overlay", "tub/num_records", "user/mode", "recording"],
        outputs=[
            "user/steering",
            "user/throttle",
            "user/mode",
            "recording",
            "web/buttons",
        ],
        threaded=True,
    )

    # uart controller
<<<<<<< HEAD
    uart = UART_backup_driver("/dev/ttyACM0")
=======
    uart = UART_backup_driver("/dev/tty.usbmodem103")
>>>>>>> 716b13a5bd616ae0c56dac041a8fd976b53cd3ca
    V.add(
        uart,
        inputs=["user/throttle", "user/steering", "critical/health_check"],
        outputs=[],
        threaded=False,
    )

    V.start(rate_hz=DRIVE_LOOP_HZ)
