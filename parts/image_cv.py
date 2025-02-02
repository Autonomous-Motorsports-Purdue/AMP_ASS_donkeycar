import cv2

class Object_Detection():
    # def __init__():

    def run(self, image):
        if image is not None:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Convert to binary - convert all pixels with values less than 127 to 0 and greater than 127 to 255
            ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

            # Detect Contours
            contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Draw Contours
            # cv2.drawContours(image, contours, -1, (0,255,0),2)
            # cv2.imshow("Contours", image)
            # cv2.waitKey(1)


            # Find Largest Contour
            maxContour = None
            maxArea = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > maxArea:
                    maxArea = area
                    maxContour = contour
            # Draw largest contour
            cv2.drawContours(image, [maxContour], 0, (255,0,0), 2)

            # Draw Contour centroid
            M = cv2.moments(maxContour)

            # Calculate the centroid of the contour
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            
            # Draw centroid
            cv2.circle(image, (cX, cY), 5, (0,0,255),-1)
            cv2.imshow("Contour", image)
            cv2.waitKey(1)

            return image, cX, cY, maxArea

            
