import vehicle
import logger

v = vehicle.Vehicle()
l = logger.LoggerPart(['Test'])

v.add(l)
v.start()
v.stop()