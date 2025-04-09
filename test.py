import cv2
from parts.lane_detect import LaneDetect

# class Image_Publisher():
#     def __init__(self):
#         self.cap = cv2.VideoCapture(0)
#         self.frame = None
#     def run(self):
#         print("running")
#         ret, self.frame = self.cap.read()
#         if self.frame is not None:
#             cv2.imshow("Image", self.frame)
#             return self.frame
            

cap = cv2.VideoCapture(0)
detector = LaneDetect()

while True:
    ret, img = cap.read()

    if not ret:  # Check if the frame was captured successfully
        print("Error: Failed to capture image")
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    midpoint_line, img_normal = detector.run(img,0,0)
    
    
    # Display the frame
    cv2.imshow("Webcam Feed", img_normal)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()