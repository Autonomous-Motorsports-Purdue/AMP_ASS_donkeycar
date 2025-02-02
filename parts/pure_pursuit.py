import math
class Pure_Pursuit():
    def __init__(self, wheelbase):
        self.wheelbase = wheelbase 
        self.speed = 50

    def run(self, target_position):
        targetx, targety = target_position 
        yaw = 0

        #if(velocity == 0):
         #   velocity = 1
        
       
        alpha = math.atan2(targety, targetx) - (yaw * (math.pi/180.0)); # Angle from rear of car to target. good.

        print(f"Debug Print 1: Angle from rear wheel to target: {(180.0 / math.pi) * alpha:.2f} degrees\n")

        Ld = math.sqrt(targetx**2 + targety**2)

        print(f"Debug Print 2: Distance from rear wheel to target: {Ld:.2f} Meters\n")


        steering_theta  = math.atan2((2*self.wheelbase*math.sin(alpha)),Ld); # Steering angle in radians.
        steering_theta *= (180.0/math.pi)
        steering_value = steering_theta /12.87 - 0.151/12.87

        #steering_value = math.atan((2*self.wheelbase*math.sin(alpha)/self.kv * velocity)); # Steering angle with velocity

        #currently returns angle in degrees. Need to convert from degrees to arbitrary steering value,
        #0-255

        return steering_value, self.speed
