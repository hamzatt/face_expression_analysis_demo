import json
import cv2
import base64
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import requests  
import gevent
import _thread
import time
import sys

from matplotlib.pyplot import figure
figure(num=None, figsize=(30, 15), dpi=80, facecolor='w', edgecolor='k')

apiKey = "9UedXuYEnilG9tYVOn6nTLhSMzrc9Bqc"
apiSecret = "iNBz-RIlYojrOimyfFGL5hchZw3LtBmm"

faceDetectURL = "https://api-us.faceplusplus.com/facepp/v3/detect"

# Set Font properties for matplotlib
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 30}

matplotlib.rc('font', **font)

def show_webcam(mirror=False):
  cam = cv2.VideoCapture(0)
  while True:
      ret_val, img = cam.read()
      if mirror: 
          img = cv2.flip(img, 1)
      cv2.imshow('my webcam', img)

      img_name = "frame.jpg"
      cv2.imwrite(img_name, img)
      # print("{} written!".format(img_name))
      
      if cv2.waitKey(1) == 27: 
          break  # esc to quit
  cv2.destroyAllWindows()

def draw_axis(img, R, t, K):
    # unit is mm
    rotV, _ = cv2.Rodrigues(R)
    points = np.float32([[100, 0, 0], [0, 100, 0], [0, 0, 100], [0, 0, 0]]).reshape(-1, 3)
    axisPoints, _ = cv2.projectPoints(points, rotV, t, K, (0, 0, 0, 0))
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()), (255,0,0), 3)
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()), (0,255,0), 3)
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[2].ravel()), (0,0,255), 3)
    return img

def detectFaceAndDrawPlot():
  # Maximize plot window
  # figManager = plt.get_current_fig_manager()
  # figManager.window.showMaximized()

  plt.gcf().clear()
  plt.ion()
  plt.show()

  while(True):
    img = open("frame.jpg","rb")

    if img:
      img_arr = cv2.imread("frame.jpg", cv2.IMREAD_UNCHANGED)
      encoded_string = base64.b64encode(img.read())

      body = {'api_key' : apiKey,
              'api_secret' : apiSecret,
              'image_base64' : encoded_string,
              'return_landmark' : 2,
              'return_attributes' : 'emotion,headpose'}

      print("Requesting")
      response = requests.post(faceDetectURL, data=body)

      if response:
        print(response)

        with open('data.json', 'w') as outfile:
          json.dump(response.json(), outfile, sort_keys=True, indent=4)

        pitch = roll = yaw = 0

        data = response.json()
        faces = response.json()['faces']
        if faces[0] != None:
          if faces[0]['landmark'] != None:
            landmarks = faces[0]['landmark']

            #img_arr = draw_axis(img_arr,np.array([pitch,roll,yaw]), np.zeros((1, 3)), np.ones((3, 3)))

            # Overlay landmarks on the image.
            for key, value in landmarks.items():
              cv2.circle(img_arr, (value['x'], value['y']), 1, (0,255,0), thickness=1, lineType=8, shift=0) 

            cv2.imshow("landmarks", img_arr) 
            
          if faces[0]['attributes'] != None:
            attributes = faces[0]['attributes']
            if attributes['emotion'] != None:
              emotion = attributes['emotion']

              plt.gcf().clear()
              plt.ion()

              plt.bar(range(len(emotion)), emotion.values(), align='center')
              plt.xticks(range(len(emotion)), emotion.keys())
              plt.ylabel('Emotion Intensity')
              plt.title('Emotion Analysis')

            if attributes['headpose'] != None:
              headpose = attributes['headpose']
              pitch = headpose['pitch_angle']
              roll = headpose['roll_angle']
              yaw = headpose['yaw_angle'] 

        if cv2.waitKey(1) == 27: 
          sys.exit(0)

def main():
  _thread.start_new_thread(show_webcam,(False,))
  detectFaceAndDrawPlot()

if __name__== "__main__":
  main()
