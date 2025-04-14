import cv2
import numpy as np
from math import factorial

class LaneDetect():
    def __init__(self):
        self.blockSizeGaus = 117
        self.constantGaus = -17
        self.closing_iterations = 1
        self.kernel_size = 3
        self.waitTime = 0
        # self.img_count = 0
        self.crop_top = 350
        self.crop_bottom = 550
        # self.image = inImage
        # self.depth = inDepth

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

    def comb(self, n, k):
        """
        Returns the combination of n choose k.
        """
        return factorial(n) / factorial(k) / factorial(n - k)

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

    def get_bezier_curve(self, contour, x_shift):
        contour_points = np.transpose(np.nonzero(contour))[0::8]
        control_points = np.array(self.get_bezier(contour_points))
        t = np.linspace(0, 1, 40)
        curve = self.plot_bezier(t, control_points)
        curve = np.flip(curve, axis=1)
        curve[:,0] += x_shift
        return curve

    def draw_bezier_curve(self, img, contour, x_shift):
        """
        Draws a bezier curve on the image.
        """
        # choose every 8th point so that the bezier curve is not too complex and it's faster
        contour_points = np.transpose(np.nonzero(contour))[0::8]
        control_points = np.array(self.get_bezier(contour_points))
        t = np.linspace(0, 1, 40)
        curve = self.plot_bezier(t, control_points)
        curve = np.flip(curve, axis=1)
        curve[:,0] += x_shift
        cv2.polylines(img, [np.int32(curve)], isClosed=False, color=(255, 255, 255), thickness=2)

        return curve
    
    def get_midpoint_control(self, sobel1, img):
        largest = self.find_two_largest_contours(img)

        contour1 = self.crop_to_contour(sobel1, largest[0])
        contour2 = self.crop_to_contour(sobel1, largest[1])

        contour_points1 = np.transpose(np.nonzero(contour1))[0::8]
        control_points1 = np.array(self.get_bezier(contour_points1))
        control_points1 = np.flip(control_points1, axis=1)
        control_points1 = control_points1 + np.array([cv2.boundingRect(largest[0])[0], self.crop_top])

        contour_points2 = np.transpose(np.nonzero(contour2))[0::8]
        control_points2 = np.array(self.get_bezier(contour_points2))
        control_points2 = np.flip(control_points2, axis=1) 
        control_points2 = control_points2 + np.array([cv2.boundingRect(largest[1])[0], self.crop_top])

        return (control_points1 + control_points2) / 2




    def find_two_largest_contours(self, contours):
        """
        Returns the two largest contours in the list of contours.
        """
        largest_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
        return largest_contours

    def crop_to_contour(self, img, contour):
        """
        Returns an image cropped to the contour.
        """
        x,y,w,h = cv2.boundingRect(contour)
        rect = np.intp(cv2.boxPoints(cv2.minAreaRect(contour)))
        crop = img[y:y+h, x:x+w]
        mask = np.zeros_like(crop)
        rect = rect - np.array([x, y])
        cv2.drawContours(mask,[rect], 0, 255, -1)
        # cv2.imshow("mask", mask)
        return cv2.bitwise_and(crop, mask)        
    def kernelx(self, x):
        """
        Returns a square kernel of size x by x.
        """
        return np.ones((x,x),np.uint8)

    def gaussian_threshold(self, img, blockSize, constant):
        """
        Returns an image thresholded using adaptive gaussian thresholding.
        """
        return cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                    cv2.THRESH_BINARY,blockSize,constant)

    def crop_image_top(self, img):
        """
        Returns an image cropped from the top.
        """
        rows, cols = img.shape[:2]
        return img[self.crop_top:self.crop_bottom, 0:cols]
    
    def convert_to_real(self, point, camera_params):
        u = point[0] # X coordinate from image
        v = point[1] # Y coordinate from image
        Z = point[2] # Depth

        # c_x = Optical center along x axis, defined in pixels
        # c_y = Optical center along y axis, defined in pixels
        # f_x = Focal length in pixels along x axis. 
        # f_y = Focal length in pixels along y axis. 
        X = ((u - camera_params['c_x']) * Z) / (camera_params['f_x'])
        Y = ((v - camera_params['c_y']) * Z) / (camera_params['f_y'])
        return [X,Y,Z]
    
    def run(self, in_image_rgb, in_image_depth, in_img_params):
        # img_normal = cv2.imread(f'imgs/img_{self.img_count}.jpg')
        # img_normal = self.conn.root.get_rgb_frame()
        img_normal = in_image_rgb
        # img_depth = self.conn.root.get_depth_frame()
        img_depth = in_image_depth
        img_params = in_img_params

        # cv2.imshow("first img normal", img_normal)
        img = cv2.cvtColor(img_normal, cv2.COLOR_BGR2GRAY)
        #img = cv2.medianBlur(img,5)

        # Crop image to reduce value range and remove sky/background
        cropped_image = self.crop_image_top(img)
        # cv2.imshow("cropped", cropped_image)
        img_normal = cv2.cvtColor(img_normal, cv2.COLOR_BGR2RGB)
        # Gaussian Thresholding
        gaussian = self.gaussian_threshold(cropped_image, self.blockSizeGaus, self.constantGaus)
        
        opening = cv2.morphologyEx(gaussian,cv2.MORPH_OPEN, self.kernelx(self.kernel_size), iterations = self.closing_iterations)
        openclose = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, self.kernelx(self.kernel_size), iterations = self.closing_iterations)


        linesP2 = cv2.HoughLinesP(openclose, 1, np.pi / 180, 50, None, minLineLength=60, maxLineGap=40)
        lines = np.zeros_like(openclose) #cv2.cvtColor(np.zeros_like(openclose), cv2.COLOR_GRAY2BGR)

        if linesP2 is not None:
            for i in range(0, len(linesP2)):
                l = linesP2[i][0]
                cv2.line(lines, (l[0], l[1]), (l[2], l[3]), (255, 255, 255), 3, cv2.LINE_AA)

        special_kernel = np.array([[0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0]], np.uint8)

        ret, lines = cv2.threshold(lines, 127, 255, cv2.THRESH_BINARY)
        lines = cv2.dilate(lines, special_kernel, iterations = 1)
        lines_dilated = cv2.bitwise_or(lines, openclose)
        # cv2.imshow("OpenCLoselines", lines)
        
        open_open = cv2.morphologyEx(lines_dilated, cv2.MORPH_OPEN, self.kernelx(3), iterations = 2)
        # cv2.imshow("OpenOpen", open_open)

        sobel1 = cv2.Sobel(open_open, cv2.CV_8UC1, 1, 0, ksize=3)

        contours_open_open,_ = cv2.findContours(open_open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # cpy_img = cropped_image.copy()

        if len(contours_open_open) >= 1:
            cv2.drawContours(cropped_image, contours_open_open, -1, (0,255,0), 5)
            for contour in contours_open_open:
                #print(cv2.contourArea(contour))        
                x,y,w,h = cv2.boundingRect(contour)
                #print("X: " , x , "\tY: ", y, "\tW: " , w ,"\tH: ", h)
                cv2.rectangle(cropped_image, (x,y), (x+w,y+h), (0,0,255), 1)

                centroid, dimensions, angle = cv2.minAreaRect(contour)
                # draw rotated rect
                # rect = cv2.minAreaRect(contour)
                # box = cv2.boxPoints(rect)
                # box = np.int0(box)
                # cv2.drawContours(cpy_img,[box],0,(0,0,255),2)

        largest = self.find_two_largest_contours(contours_open_open)

        if len(largest) == 0:
            return np.array([0,0]), img_normal

        contour1 = self.crop_to_contour(sobel1, largest[0])
        if len(largest) == 1:
            print("Only one contour")
            curve1 = self.get_bezier_curve(contour1, cv2.boundingRect(largest[0])[0])
            return curve1, img_normal

        contour2 = self.crop_to_contour(sobel1, largest[1])

        curve1 = self.get_bezier_curve(contour1, cv2.boundingRect(largest[0])[0])
        curve2 = self.get_bezier_curve(contour2, cv2.boundingRect(largest[1])[0])
    

        midpoint_line = (curve1 + curve2) / 2

        height_adjust_midpoint_line = midpoint_line.copy()

        height_adjust_midpoint_line[:, 1] += self.crop_top



        cv2.polylines(img_normal, [np.int32(height_adjust_midpoint_line )], isClosed=False, color=(255, 255, 255), thickness=2)
        # cv2.imshow("test", img_normal)
        # cv2.waitKey(10)


        return midpoint_line, img_normal
