import numpy as np
import cv2
from math import factorial

class Process_o():
    def __init__(self):
        pass

    def run(self, lanes, track):
        x, y = track.shape[1], track.shape[0]

        track = cv2.resize(track, (x, y))
        
        sobel, curve = self.track(track)
        return sobel, curve
    
    def get_bezier_curve(self, line):
        points = np.transpose(np.nonzero(line))[0::8]
        control_points = np.array(self.get_bezier(points))
        t = np.linspace(0, 1, 40)
        curve = self.plot_bezier(t, control_points)
        curve = np.flip(curve, axis=1)
        return curve

    def plot_bezier(self, t, cp):
        """
        Plots a bezier curve.
        t is the time values for the curve.
        cp is the control points of the curve.
        return is a tuple of the x and y values of the curve.
        """
        cp = np.array(cp)
        num_points, d = np.shape(cp)   # Number of points, Dimension of points
        num_points = num_points - 1
        curve = np.zeros((len(t), d))
        
        for i in range(num_points+1):
            # Bernstein polynomial
            val = self.comb(num_points,i) * t**i * (1.0-t)**(num_points-i)
            curve += np.outer(val, cp[i])
        
        return curve
    
    def get_bezier(self, points):
        """
        Returns the control points of a bezier curve.
        """
        num_points = len(points)

        x, y = points[:,0], points[:,1]

        # bezier matrix for a cubic curve
        bezier_matrix = np.array([[-1, 3, -3, 1,], [3, -6, 3, 0], [-3, 3, 0, 0], [1, 0, 0, 0]])
        bezier_inverse = np.linalg.inv(bezier_matrix)

        normalized_length = self.normalize_path_length(points)

        points_matrix = np.zeros((num_points, 4))

        for i in range(num_points):
            points_matrix[i] = [normalized_length[i]**3, normalized_length[i]**2, normalized_length[i], 1]

        points_transpose = points_matrix.transpose()
        square_points = np.matmul(points_transpose, points_matrix)

        square_inverse = np.zeros_like(square_points)

        if (np.linalg.det(square_points) == 0):
            print("Uninvertible matrix")
            square_inverse = np.linalg.pinv(square_points)
        else:
            square_inverse = np.linalg.inv(square_points)

        # solve for the solution matrix
        solution = np.matmul(np.matmul(bezier_inverse, square_inverse), points_transpose)

        # solve for the control points
        control_points_x = np.matmul(solution, x)
        control_points_y = np.matmul(solution, y)

        return list(zip(control_points_x, control_points_y))
    
    def normalize_path_length(self, points):
            """
            Returns a list of the normalized path length of the points.
            """
            path_length = [0]
            x, y = points[:,0], points[:,1]

            # calculate the path length
            for i in range(1, len(points)):
                path_length.append(np.sqrt((x[i] - x[i - 1])**2 + (y[i] - y[i - 1])**2) + path_length[i - 1])
            
            # normalize the path length
            # computes the percentage of path length at each point
            pct_len = []
            for i in range(len(path_length)):
                if (path_length[i] == 0):
                    pct_len.append(0.01)
                    continue
                pct_len.append(path_length[i] / path_length[-1])
            
            return pct_len
    
    def comb(self, n, k):
        """
        Returns the combination of n choose k.
        """
        return factorial(n) / factorial(k) / factorial(n - k)
    
    def track(self, img):
        kernel5 = np.ones((5,5),np.uint8)
        kernel3 = np.ones((3,3),np.uint8)
        # erode = cv2.erode(img, kernel5, iterations = 5)
        # img = cv2.dilate(erode, kernel5, iterations = 5)

        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel5, iterations = 5)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel5, iterations = 2)
        #img = cv2.GaussianBlur(img, (5, 5), 0)


        contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        
        if len(contours) == 0:
            print("No contours found")
            return np.zeros_like(img)

        # Select the largest contour
        c = max(contours, key = cv2.contourArea)

        # Draw the largest contour
        x,y,w,h = cv2.boundingRect(c)
        rect = np.intp(cv2.boxPoints(cv2.minAreaRect(c)))
        mask = np.zeros_like(img)
        cv2.drawContours(mask, [rect], -1, (255), -1)
        cv2.imshow("mask", mask)
        img = cv2.bitwise_and(img, mask)


        mask = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel5, iterations = 15)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel5, iterations = 2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel3, iterations = 2)

        img = cv2.bitwise_and(img, mask)


        inverse = cv2.bitwise_not(img)
        sobelLeft = cv2.Sobel(img, cv2.CV_8UC1, 1, 1, ksize=5)
        sobelRight = cv2.Sobel(inverse, cv2.CV_8UC1, 1, 1, ksize=5)

        leftCurve = self.get_bezier_curve(sobelLeft)
        rightCurve = self.get_bezier_curve(sobelRight)

        mid = (leftCurve + rightCurve) / 2

        print(mid)

        curveImage = np.zeros_like(img)
        cv2.polylines(curveImage, [np.int32(leftCurve)], isClosed=False, color=(255), thickness=2)
        cv2.polylines(curveImage, [np.int32(rightCurve)], isClosed=False, color=(255), thickness=2)

        sobel = cv2.bitwise_or(sobelLeft, sobelRight)

        cv2.imshow("sobel", sobel)
        cv2.imshow("curve", curveImage)


        return sobel, curveImage
