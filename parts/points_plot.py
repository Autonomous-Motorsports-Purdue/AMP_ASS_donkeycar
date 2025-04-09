import matplotlib.pylab as plt


class Points_plot():
    def __init__(self):
        plt.ion()
        self.fig, self.ax = plt.subplots()
    
    def run(self,points):
        if points is None or len(points) == 0:
            return
        
        self.ax.cla()
        x = points[:,0]
        y = points[:,1]
        self.ax.plot(x, y, marker='o')
        self.ax.set_title("Real-Time Point Plot")
        self.ax.set_xlim(left=-5, right=5)
        self.ax.set_ylim(bottom=-5,top=5)  # Ensures the x-axis includes 0
        self.ax.grid(True)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()