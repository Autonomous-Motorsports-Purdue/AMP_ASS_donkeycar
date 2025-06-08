class Control_Muxer:
    def __init__(self):
        pass # takes in nothing lol

    def run(self, user_steer, user_throt, auto_steer, auto_throt):
        if user_steer is None or user_throt is None:
            return auto_steer, auto_throt
        if user_steer != 0 or user_throt != 0:
            print(f"USER CONTROL: {user_steer}, {user_throt}")
            return user_steer, user_throt * .7
        else:
            return auto_steer, auto_throt

    def shutdown(self):
        pass
