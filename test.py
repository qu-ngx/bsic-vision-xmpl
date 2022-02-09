# # Import the camera server
# from cscore import CameraServer

# Import OpenCV and NumPy
import cv2 as cv
import numpy as np
import time

  
cap = cv.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    frame = cv.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
    cv.imshow('Input', frame)

    c = cv.waitKey(1)
    if c == 27:
        break

cap.release()



def main():
   config = frame
   camera = config['cameras'][0]

   width = camera['width']
   height = camera['height']



   # Table for vision output information
   vision_nt = frame.getTable('Vision') #NetworkTables

   # Allocating new images is very expensive, always try to preallocate
   img = np.zeros(shape=(240, 320, 3), dtype=np.uint8)

   # Wait for NetworkTables to start
   time.sleep(0.5)

   while True:
      start_time = time.time()

      frame_time, input_img = frame.grabFrame(img)
      output_img = np.copy(input_img)

      # Notify output of error and skip iteration
      if frame_time == 0:
         output_stream.notifyError(input_stream.getError())
         continue

      # Convert to HSV and threshold image
      hsv_img = cv.cvtColor(input_img, cv2.COLOR_BGR2HSV)
      binary_img = cv.inRange(hsv_img, (65, 65, 200), (85, 255, 255))

      _, contour_list, _ = cv.findContours(binary_img, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

      x_list = []
      y_list = []

      for contour in contour_list:

         # Ignore small contours that could be because of noise/bad thresholding
         if cv.contourArea(contour) < 15:
            continue

         cv.drawContours(output_img, contour, -1, color = (255, 255, 255), thickness = -1)

         rect = cv.minAreaRect(contour)
         center, size, angle = rect
         center = tuple([int(dim) for dim in center]) # Convert to int so we can draw

         # Draw rectangle and circle
         cv.drawContours(output_img, [cv.boxPoints(rect).astype(int)], -1, color = (0, 0, 255), thickness = 2)
         cv.circle(output_img, center = center, radius = 3, color = (0, 0, 255), thickness = -1)

         x_list.append((center[0] - width / 2) / (width / 2))
         x_list.append((center[1] - width / 2) / (width / 2))

      vision_nt.putNumberArray('target_x', x_list)
      vision_nt.putNumberArray('target_y', y_list)

      processing_time = time.time() - start_time
      fps = 1 / processing_time
      cv.putText(output_img, str(round(fps, 1)), (0, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
      output_stream.putFrame(output_img)

main()
