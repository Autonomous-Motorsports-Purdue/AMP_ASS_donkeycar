import donkeycar as dk
from donkeycar.parts.controller import LocalWebController

import time
from parts.health_check import HealthCheck
from parts.uart import UART_Driver
from constants import DRIVE_LOOP_HZ

if __name__ == "__main__":
    # web controller
    V = dk.vehicle.Vehicle()
    health_check = HealthCheck("localhost", 6000)
    V.add(health_check, inputs=None, outputs=["critical/health_check"])

    controller = LocalWebController()
    # web controller just expects all these things even though they don't exist
    V.add(
        controller,
        inputs=["cam/image_array", "tub/num_records", "user/mode", "recording"],
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
    uart = UART_Driver("/dev/ttys011")
    V.add(
        uart,
        inputs=["user/throttle", "user/steering", "critical/health_check"],
        outputs=[],
        threaded=False,
    )

    V.start(rate_hz=DRIVE_LOOP_HZ)
