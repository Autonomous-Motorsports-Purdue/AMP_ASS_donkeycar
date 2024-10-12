import donkeycar as dk
from donkeycar.parts.controller import LocalWebController

import time
from util.uart import UART_Driver

if __name__ == "__main__":
    # web controller
    V = dk.vehicle.Vehicle()

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
    uart = UART_Driver("COM5")
    V.add(uart, inputs=["user/throttle", "user/steering"], outputs=[], threaded=False)

    V.start(rate_hz=20)
