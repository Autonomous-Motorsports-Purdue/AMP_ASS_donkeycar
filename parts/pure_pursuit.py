import math
class Pure_Pursuit():
    def __init__(self, speed):
        self.wheelbase = 1.000506 
        self.speed = speed

    def run(self, target_position):
        targetx, targety, _ = target_position 
        targety = -targety
        print(f"{targetx}, {targety}")
        #targetx, targety = 3, -2.5 
        # we move forward in x 
        yaw = 0

        #if(velocity == 0):
         #   velocity = 1
        
       
        alpha = math.atan2(targety, targetx) - (yaw * (math.pi/180.0)); # Angle from rear of car to target. good.


        #print(f"Debug Print 1: Angle from rear wheel to target: {(180.0 / math.pi) * alpha:.2f} degrees\n")

        Ld = math.sqrt(targetx**2 + targety**2)

        print(f"Debug Print 2: Distance from rear wheel to target: {Ld:.2f} Meters\n")


        steering_theta  = math.atan2((2*self.wheelbase*math.sin(alpha)),Ld); # Steering angle in radians.
        steering_theta *= (180.0/math.pi)
        # if turning left, use left coeff. If using right, turn right.
        '''
        if steering_theta > 0: # left
            steering_value = steering_theta / 23.644
        else:
            steering_value = steering_theta / 22.096
        '''
        #steering_value = steering_theta /18.523 
        steering_value = steering_theta /15.523 

        print(f"Sterring theta: {steering_theta}\tSteering value: {steering_value}")
        #steering_value = math.atan((2*self.wheelbase*math.sin(alpha)/self.kv * velocity)); # Steering angle with velocity

        #currently returns angle in degrees. Need to convert from degrees to arbitrary steering value,
        #0-255

        return steering_value, self.speed
