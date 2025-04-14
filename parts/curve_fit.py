import numpy as np
import cv2
from math import factorial
from scipy.spatial import KDTree, distance_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from scipy.sparse import csr_matrix

class Curve_fit():
    def __init__(self):
        pass

    def run(self, lanes, track):
        x, y = track.shape[1], track.shape[0]
        
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

    def mst_clustering_largest_component(self, binary_image, distance_threshold=10):
        # Step 1: Extract coordinates of foreground pixels
        points = np.column_stack(np.nonzero(binary_image))
        
        if len(points) == 0:
            return np.zeros_like(binary_image, dtype=bool)  # empty image

        # Step 2: Compute distance matrix
        dist_matrix = distance_matrix(points, points)

        # Step 3: Compute MST
        mst = minimum_spanning_tree(dist_matrix)

        # Step 4: Convert MST to a graph and cut edges longer than threshold
        mst = mst.toarray()
        mst[mst > distance_threshold] = 0  # cut long edges

        # Step 5: Use connected components on the modified MST
        graph = csr_matrix((mst + mst.T) > 0)  # symmetrize
        n_components, labels = connected_components(graph)

        # Step 6: Find largest cluster
        unique, counts = np.unique(labels, return_counts=True)
        largest_cluster_label = unique[np.argmax(counts)]

        # Step 7: Mask the original image
        mask = labels == largest_cluster_label
        largest_cluster_coords = points[mask]

        # Step 8: Create output image
        output_image = np.zeros_like(binary_image, dtype=bool)
        output_image[largest_cluster_coords[:, 0], largest_cluster_coords[:, 1]] = True

        return output_image

    
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
        img = cv2.bitwise_and(img, mask)

        mask = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel3, iterations = 3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel5, iterations = 3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel3, iterations = 2)

        #img = cv2.bitwise_and(img, mask)
        img = mask

        sobelLeft = cv2.Sobel(img, cv2.CV_8UC1, 1, 0, ksize=1)
        sobelLeft[:, :2] = 0
        left_cluster = self.mst_clustering_largest_component(sobelLeft, 75)

        # radius = 30
        # min_neighbors = 30
        
        # rows, cols = np.nonzero(sobelLeft) #Use scipy to get rid of outliers
        # points = np.column_stack((rows, cols))
        # tree = KDTree(points)
        # neighbor_counts = tree.query_ball_point(points, radius)
        # inlier_mask = np.array([len(neighbors) >= min_neighbors for neighbors in neighbor_counts])
        # if inlier_mask.size != 0:
        #     sobelLeft[rows[~inlier_mask], cols[~inlier_mask]] = 0

        inverse = cv2.bitwise_not(img)
        sobelRight = cv2.Sobel(inverse, cv2.CV_8UC1, 1, 0, ksize=1)
        sobelRight[:, -2:] = 0
        right_cluster = self.mst_clustering_largest_component(sobelRight, 75)
        # clustered = clustered.astype(np.uint8) * 255

        # difference = cv2.absdiff(sobelRight, clustered)
        # difference = cv2.dilate(difference, kernel3, iterations = 1)

        # clustered = cv2.dilate(clustered, kernel3, iterations = 1)
        
        # # Turn clusterd into an rgb image with difference as red channel
        # clustered = cv2.cvtColor(clustered, cv2.COLOR_GRAY2BGR)
        
        # clustered[:,:,0] = difference




        # cv2.imshow("clustered", clustered)

        
        # rows, cols = np.nonzero(sobelRight) #Use scipy to get rid of outliers
        # points = np.column_stack((rows, cols))
        # tree = KDTree(points)
        # neighbor_counts = tree.query_ball_point(points, radius)
        # inlier_mask = np.array([len(neighbors) >= min_neighbors for neighbors in neighbor_counts])
        # if inlier_mask.size != 0:
        #     sobelRight[rows[~inlier_mask], cols[~inlier_mask]] = 0
        
        leftCurve = self.get_bezier_curve(sobelLeft)
        rightCurve = self.get_bezier_curve(sobelRight)

        cluster_curve_left = self.get_bezier_curve(left_cluster)
        cluster_curve_right = self.get_bezier_curve(right_cluster)

        cluster_curve_image =np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
        cv2.polylines(cluster_curve_image, [np.int32(cluster_curve_left)], isClosed=False, color=(255,0,0), thickness=2)
        cv2.polylines(cluster_curve_image, [np.int32(cluster_curve_right)], isClosed=False, color=(0,255,0), thickness=2)
        cv2.imshow("cluster_curve", cluster_curve_image)

        mid = (leftCurve + rightCurve) / 2

        #Curve image is an empty rgb image
        x, y = img.shape[1], img.shape[0]
        curveImage = np.zeros((y, x, 3), np.uint8)

        cv2.polylines(curveImage, [np.int32(leftCurve)], isClosed=False, color=(255,0,0), thickness=2)
        cv2.polylines(curveImage, [np.int32(rightCurve)], isClosed=False, color=(0,255,0), thickness=2)

        # Sobel is a an empty rgb image
        sobel = np.zeros((y, x, 3), np.uint8)
        sobel[:,:,0] = sobelLeft
        # Add sobel right as green channel
        sobel[:,:,1] = sobelRight
        cv2.imshow("sobel", sobel)
        cv2.imshow("curve", curveImage)


        return sobel, curveImage
