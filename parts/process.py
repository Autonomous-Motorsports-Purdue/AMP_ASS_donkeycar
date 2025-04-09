import numpy as np
import cv2
from math import factorial
from scipy.spatial import cKDTree
from scipy.interpolate import PchipInterpolator
import time

class Process():
    def __init__(self):
        pass

    def run(self, lanes, track,):
        x, y = track.shape[1], track.shape[0]

        track = cv2.resize(track, (x, y))
        
        sobel, curve, mid = self.track(track)
        return sobel, curve,mid
    
    def get_bezier_curve(self, line):
        points = np.transpose(np.nonzero(line))
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
        num_points_x, d = np.shape(cp)   # Number of points, Dimension of points
        num_points_x = num_points_x - 1
        curve = np.zeros((len(t), d))
        
        for i in range(num_points_x+1):
            # Bernstein polynomial
            val = self.comb(num_points_x,i) * t**i * (1.0-t)**(num_points_x-i)
            curve += np.outer(val, cp[i])
        
        return curve
    
    def get_bezier(self, points):
        """
        Returns the control points of a bezier curve.
        """
        num_points_x = len(points)

        x, y = points[:,0], points[:,1]

        # bezier matrix for a cubic curve
        bezier_matrix = np.array([[-1, 3, -3, 1,], [3, -6, 3, 0], [-3, 3, 0, 0], [1, 0, 0, 0]])
        bezier_inverse = np.linalg.inv(bezier_matrix)

        normalized_length = self.normalize_path_length(points)

        points_matrix = np.zeros((num_points_x, 4))

        for i in range(num_points_x):
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
    
    def get_points(self, img):
        white_points = cv2.findNonZero(img)
        
        return img, white_points
        
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
        height_crop = 200
        sobel_cropped = sobel[:-height_crop,:]
        H, W = sobel_cropped.shape[:2]
        H = H // (num_points - 1)
        slices_ = []
        points = np.zeros((num_points, 2), dtype=int)
        
        for height in range(num_points - 1):
            slices_.append(sobel_cropped[H * height:H * (height + 1), :]) 
            
        found = 0
        for idx,slice_ in enumerate(slices_):
            #cv2.imshow(f"slice{idx}",slice_)
            white_points = cv2.findNonZero(slice_)
            if white_points is not None:
                found += 1
                white_points = white_points.reshape(-1,2)
                mean_point = np.mean(white_points, axis=0)
                points[idx] = (int(mean_point[0]), int(mean_point[1]) + H * (idx-1))

        #print(points)
    
        filtered_points = points #self.distance_filter(points, 100 + 50*(8-found), 15)
        new_image = self.points_to_image(filtered_points,(720,1280))
        return new_image, filtered_points

    def get_bSpline_curve(self, sobel): #BROKEN :(
        H, W = sobel.shape[0:2]
        points = cv2.findNonZero(sobel)
        if points is None:
            return self.points_to_image([], (H, W))
        points = points.reshape(-1, 2)

        # Calculate arc-length for parameterization
        deltas = np.diff(points, axis=0)
        dist = np.hypot(deltas[:, 0], deltas[:, 1])  # Euclidean distance
        t = np.concatenate(([0], np.cumsum(dist)))

        # Normalize parameter to [0, 1]
        t = t / t[-1] if t[-1] != 0 else t
        
        # Fit cubic splines using the arc-length parameter
        spline_x = PchipInterpolator(t, points[:, 0])
        spline_y = PchipInterpolator(t, points[:, 1])

        # Uniformly sample along the normalized parameter
        t_sample = np.linspace(0, 1, 100)

        points_x = spline_x(t_sample)
        points_y = spline_y(t_sample)

        # Combine and round the spline points
        spline_points = np.vstack((points_x, points_y)).T
        spline_points = np.round(spline_points).astype(np.int32)

        print(spline_points)
        
        return spline_points
        
    def points_to_image(self,points,img_size):
        
        image = np.zeros(img_size, dtype=np.uint8)

        for x, y in points:
            x,y = int(x), int(y)
            cv2.line(image, (x, y),(x , y),color=255,thickness=1)
        return image
    
    def average_window(self, img, num_windows):
        # Find non-zero points and reshape for easier processing
        points = cv2.findNonZero(img)
        if points is None:
            return self.points_to_image([], (720, 1280))
        
        points = points.reshape(-1, 2)
        
        # Calculate min and max Y values
        max_Y = np.max(points[:, 1])
        min_Y = np.min(points[:, 1])
        
        # Define window edges using linspace
        window_edges = np.linspace(min_Y, max_Y, num_windows + 1)
        
        avg_points = []
        
        for i in range(num_windows):
            y_min = window_edges[i]
            y_max = window_edges[i + 1]
            
            # Filter points within the current window
            window_points = points[(points[:, 1] >= y_min) & (points[:, 1] < y_max)]
            
            if window_points.size > 0:
                mean_point = np.mean(window_points, axis=0)
                avg_points.append(mean_point.tolist())
            else:
                avg_points.append([-1, -1])
        
        return self.points_to_image(avg_points[1:], (720, 1280))
            
    def distance_filter(self,img, min_distance,group):
        points = cv2.findNonZero(img)
        filtered_points = []
        
        points = points.reshape(-1,2)

        for i, point in enumerate(points):
            count = 0
            for j, other_point in enumerate(points):
                if i != j:  # Skip comparing the point to itself
                    dist = np.linalg.norm(point - other_point)  # Using np.linalg.norm for distance
                    if dist <= min_distance:
                        count += 1
                        if count > group:
                            filtered_points.append(point)
                            continue
            # If the point is sufficiently distant from at least two other points, keep it
            
        
        return self.points_to_image(filtered_points,(720,1280))
    
    def optimized_filter(self, img, min_distance, group):
        # Find non-zero points and reshape for easier processing
        points = cv2.findNonZero(img)
        if points is None:
            return self.points_to_image([], (720, 1280))

        points = points.reshape(-1, 2)
        filtered_points = []

        # Use KD-Tree for efficient distance queries
        tree = cKDTree(points)

        for i, point in enumerate(points):
            # Query for neighbors within the min_distance (including itself)
            indices = tree.query_ball_point(point, min_distance)
            
            # Subtract one if the point includes itself in the count
            if len(indices) - 1 >= group:
                filtered_points.append(point)

        # Convert to list if not empty, else return empty list
        filtered_points = np.array(filtered_points).tolist() if filtered_points else []

        return self.points_to_image(filtered_points, (720, 1280))
    
    def fill_lower(self,img,range):
        H,W = img.shape[:2]
        
        img = img[:-range,:]
        
        img = np.vstack((img, np.full((range,W),255, dtype = img.dtype)))
        
        return img
           
    def track(self, img):
        
        kernel5 = np.ones((5,5),np.uint8)
        kernel3 = np.ones((3,3),np.uint8)
        # erode = cv2.erode(img, kernel5,) iterations = 5)
        # img = cv2.dilate(erode, kernel5, iterations = 5)
        
        img_morph = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel5, iterations = 5)
        img_morph = cv2.morphologyEx(img_morph, cv2.MORPH_CLOSE, kernel5, iterations = 2)
        #img = cv2.GaussianBlur(img, (5, 5), 0)

        contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        
        if len(contours) == 0:
            print("No contours found")
            return np.zeros_like(img_morph)

        # Select the largest contour
        c = max(contours, key = cv2.contourArea)

        # Draw the largest contour
        x,y,w,h = cv2.boundingRect(c)
        rect = np.intp(cv2.boxPoints(cv2.minAreaRect(c)))
        mask = np.zeros_like(img)
        cv2.drawContours(mask, [rect], -1, (255), -1)
        img = cv2.bitwise_and(img, mask)
        img = self.fill_lower(img,200)
        
        mask = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel3, iterations = 5)
        mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE, kernel3, iterations= 5)
        mask = cv2.GaussianBlur(mask,(5,5),0)
        mask[mask != 0] = 255
        
        cv2.imshow("mask",mask)

        #img = cv2.bitwise_and(img, mask)
        inverse = cv2.bitwise_not(mask)
        sobelLeft = cv2.Sobel(inverse, cv2.CV_8UC1, 1, 0, ksize=3)
        sobelRight = cv2.Sobel(mask, cv2.CV_8UC1, 1, 0, ksize=3)
        
        #remove outliers
        sobelLeft = self.optimized_filter(sobelLeft,200,70)
        sobelRight = self.optimized_filter(sobelRight,200,70)
        
        
        cv2.imshow('filteredL',sobelLeft)
        cv2.imshow('filteredR',sobelRight)
        
        #get average points from these two presented curves
        sobelLeft = self.average_window(sobelLeft,20)
        sobelRight = self.average_window(sobelRight,20)
        
        #show images for debugging
        cv2.imshow("left_points",sobelLeft)
        cv2.imshow("right_points",sobelRight)
        
        leftCurve = self.get_bezier_curve(sobelLeft) #leftCurve = cv2.findNonZero(sobelLeft)
        rightCurve = self.get_bezier_curve(sobelRight) #rightCurve = cv2.findNonZero(sobelRight)  

        mid = (leftCurve + rightCurve) / 2

        curveImageR = np.zeros_like(img)
        curveImageL = np.zeros_like(img)
        curveImageM = np.zeros_like(img)
        
        cv2.polylines(curveImageL, [np.int32(leftCurve)], isClosed=False, color=(255), thickness=2)
        cv2.polylines(curveImageR, [np.int32(rightCurve)], isClosed=False, color=(255), thickness=2)
        cv2.polylines(curveImageM, [np.int32(mid)], isClosed=False, color=(255), thickness=2)
        
        sobel = cv2.bitwise_or(sobelLeft,sobelRight)
        curveImage = cv2.bitwise_or(curveImageL,curveImageR)
        curveImage =  cv2.bitwise_or(curveImage,curveImageM)
        
        cv2.imshow("sobel", sobel)
        cv2.imshow("curveL",curveImage)
        
        
        curveImageM = self.average_window(curveImageM,10)

        return sobel, curveImage, curveImageM   
