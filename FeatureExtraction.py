from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import matplotlib.pyplot as plt
import argparse
import imutils
import time
import dlib
import cv2
import reference_world as world

def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
  A = dist.euclidean(eye[1], eye[5])
  B = dist.euclidean(eye[2], eye[4])
    
    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
  C = dist.euclidean(eye[0], eye[3])
    
    # compute the eye aspect ratio
  ear = (A + B) / (2.0 * C)
    
    # return the eye aspect ratio
  return ear
  
def mouth_aspect_ratio(mouth):

  A = dist.euclidean(mouth[1], mouth[7])
  B = dist.euclidean(mouth[2], mouth[6])
  C = dist.euclidean(mouth[3], mouth[5])
  D = dist.euclidean(mouth[0], mouth[4])
  mar = (A + B + C) / (3.0 * D)
  return mar
  
  
  
  
def head_angle(landmark,image)
  refImgPts = world.ref2dImagePoints(landmark)
  height, width, channels = image.shape
        focalLength = args["focal"] * width
        cameraMatrix = world.cameraMatrix(focalLength, (height / 2, width / 2))
        mdists = np.zeros((4, 1), dtype=np.float64)
        
  # calculate rotation and translation vector using solvePnP
  success, rotationVector, translationVector = cv2.solvePnP(face3Dmodel, refImgPts, cameraMatrix, mdists)
  noseEndPoints3D = np.array([[0, 0, 1000.0]], dtype=np.float64)
  noseEndPoint2D, jacobian = cv2.projectPoints(noseEndPoints3D, rotationVector, translationVector, cameraMatrix, mdists)

  #  draw nose line
  p1 = (int(refImgPts[0, 0]), int(refImgPts[0, 1]))
  p2 = (int(noseEndPoint2D[0, 0, 0]), int(noseEndPoint2D[0, 0, 1]))
  cv2.line(frame, p1, p2, (110, 220, 0),thickness=2, lineType=cv2.LINE_AA)

  # calculating euler angles
  rmat, jac = cv2.Rodrigues(rotationVector)
  angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
  Thetay = np.arctan2(-Qy[2][0], np.sqrt((Qy[2][1] * Qy[2][1] ) + (Qy[2][2] * Qy[2][2])))
  return ThetaY
  


def getFrame(sec):
    start = 180000
    vidcap.set(cv2.CAP_PROP_POS_MSEC, start + sec*1000)
    hasFrames,image = vidcap.read()
    return hasFrames, image
    

data = []
angles = []
for j in [60]:
  for i in [10]: #10minutes
    vidcap = cv2.VideoCapture(' ')
    sec = 0
    frameRate = 1
    success, image  = getFrame(sec)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rect = detector(gray, 1)
    landmarks = predictor(image=gray, box=rect)

    count = 0
    while success and count < 300: #extract 300 frame each video
          landmarks = predictor(image=gray, box=rect)
          if sum(sum(landmarks)) != 0:
              count += 1
              data.append(landmarks)
              
              headangle = head_angle(landmarks,image)
              angles.append(headangle)
              
              sec = sec + frameRate
              sec = round(sec, 2)
              success, image = getFrame(sec)
              print(count)
              
          else:
              sec = sec + frameRate
              sec = round(sec, 2)
              success, image = getFrame(sec)
              print("not detected")


data = np.array(data)
angles = np.array(angles)

features = []
for d and angle in data and angles :
  eye = d[37:48]
  mouth = d[49,68]
  ear = eye_aspect_ratio(eye)
  mar = mouth_aspect_ratio(mouth)
  features.append([ear, mar, angle])
