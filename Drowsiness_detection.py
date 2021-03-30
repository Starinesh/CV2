# import the necessary packages
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

def moving_average(x, w):
  return np.convolve(x, np.ones(w), 'valid') / w
    
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

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
    help="path to facial landmark predictor")
ap.add_argument("-f", "--focal",type=float,
                    help="Callibrated Focal Length of the camera")
ap.add_argument("-w", "--webcam", type=int, default=0,
    help="index of webcam on system")
args = vars(ap.parse_args())

face3Dmodel = world.ref3DModel()
EYE_AR_THRESH = 0.25
COUNTER = 0
eye_state = []
mouth_state = []
times = []

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(Start, End) = face_utils.FACIAL_LANDMARKS_IDXS["inner_mouth"]

# start the video stream thread
print("[INFO] starting video stream thread...")
#vs = VideoStream(src=args["webcam"]).start()
#time.sleep(1.0)



cap = cv2.VideoCapture("nonsleepyCombination.avi")
# loop over frames from the video stream
while True:
    #frame = vs.read()
    ret, frame = cap.read()
    if ret is False:
     break
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # detect faces in the grayscale frame
    rects = detector(gray, 0)

# loop over the face detections
    for rect in rects:
        
        x1 = rect.left() # left point
        y1 = rect.top() # top point
        x2 = rect.right() # right point
        y2 = rect.bottom() # bottom point
        # Draw a rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        landmarks = predictor(image=gray, box=rect)
        
        refImgPts = world.ref2dImagePoints(landmarks)
        height, width, channels = frame.shape
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
        
        
        print('*' * 80)
            # print(f"Qx:{Qx}\tQy:{Qy}\tQz:{Qz}\t")
        x = np.arctan2(Qx[2][1], Qx[2][2])
        y = np.arctan2(-Qy[2][0], np.sqrt((Qy[2][1] * Qy[2][1] ) + (Qy[2][2] * Qy[2][2])))
        z = np.arctan2(Qz[0][0], Qz[1][0])
            # print("ThetaX: ", x)
        print("ThetaY: ", y)
            # print("ThetaZ: ", z)
        print('*' * 80)
        
        if angles[1] < 35:
           GAZE = "Looking: Left"
        elif angles[1] > 65 :
           GAZE = "Looking: Right"
        else:
           GAZE = "Forward"

        cv2.putText(frame, GAZE, (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 80), 2)



        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        
        # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0
        
        Mouth = shape[Start:End]
        MAR = mouth_aspect_ratio(Mouth)
        mar = MAR
                        
        

# compute the convex hull for the left and right eye, then
        # visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        MouthHull = cv2.convexHull(Mouth)
        cv2.drawContours(frame, [MouthHull], -1, (0, 255, 0), 1)
# draw the computed eye aspect ratio on the frame to help
        # with debugging and setting the correct eye aspect ratio
        # thresholds and frame counters
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (0, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "MAR: {:.2f}".format(mar), (0, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        eye_state.append(ear)
        mouth_state.append(mar)
        
               # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
 
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

#Eye_state = moving_average(eye_state, 40)
#Mouth_state = moving_average(mouth_state, 40)

#time1 = [x for x in range(len(Eye_state)) ]
#time2 = [x for x in range(len(Mouth_state)) ]

#fig, (ax1, ax2) = plt.subplots(1, 2)
#ax1.plot(time1,Eye_state)
#ax1.set_ylabel('eye_aspect_ratio')
#ax1.set_xlabel('frame')
#ax2.plot(time2,Mouth_state)
#ax2.set_ylabel('mouth_aspect_ratio')
#ax2.set_xlabel('frame')
#plt.show()
        
        

cv2.destroyAllWindows()
#vs.stop()

#python Drowsiness_detection.py --shape-predictor shape_predictor_68_face_landmarks.dat --focal 20
