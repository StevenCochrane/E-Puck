from ePuck import ePuck
import sys
import numpy as np
import cv2
import cv2.aruco as aruco
import math
import time
import matplotlib.pyplot as plt

# CALCULATE ROTATION ANGLES

def rotationMatrixToEulerAngles(R):

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])

# COMMUNICATE VELOCITY COMMANDS

def setvel(v, w):

    L = 0.053  # Axel Width
    R = 0.0205  # Wheel Radius

    vr1 = (2 * v[0] + w[0] * L) / (2 * R)
    vl1 = (2 * v [0]- w[0] * L) / (2 * R)
    rs1 = vr1 / 0.00628
    ls1 = vl1 / 0.00628

    vr2 = (2 * v[1] + w[1] * L) / (2 * R)
    vl2 = (2 * v[1] - w[1] * L) / (2 * R)
    rs2 = vr2 / 0.00628
    ls2 = vl2 / 0.00628

    vr3 = (2 * v[2] + w[2] * L) / (2 * R)
    vl3 = (2 * v[2] - w[2] * L) / (2 * R)
    rs3 = vr3 / 0.00628
    ls3 = vl3 / 0.00628

    robot1.set_motors_speed(ls1, rs1)
    robot1.step()
    robot2.set_motors_speed(ls2, rs2)
    robot2.step()
    robot3.set_motors_speed(ls3, rs3)
    robot3.step()

# LIST OF E-PUCKS : CHANGE MAC ADDRESSES : DEVICE MANAGER

epucks = {
    '3303': '10:00:e8:c5:64:37',
    '3214': '10:00:e8:c5:64:56',
    '3281': '10:00:e8:c5:61:82',
    '3276': '10:00:e8:c5:61:43',
    '3109': '10:00:e8:ad:78:1d'
}

# MISCELLANEOUS

def log(text):
    blue = '\033[1;34m'
    off = '\033[1;m'
    print(''.join((blue, '[Log] ', off, str(text))))

def error(text):
    red = '\033[1;31m'
    off = '\033[1;m'
    print(''.join((red, '[Error] ', off, str(text))))

# 0: Default Camera  1: Secondary Camera

cap = cv2.VideoCapture(0)

# READ IN CALIBRATION FILE

cv_file = cv2.FileStorage("calib.yaml", cv2.FILE_STORAGE_READ)

mtx = cv_file.getNode("camera_matrix").mat()
dist = cv_file.getNode("dist_coeff").mat()

# SET MARKER LENGTH IN (M)

markerLength=0.08

print('Connecting with the ePuck')

# CONNECT WITH E-PUCKS - ALL LEDS LIGHT WHEN CONNECTED

try:
    robot1 = ePuck(epucks['3281'])
    robot2 = ePuck(epucks['3276'])
    robot3 = ePuck(epucks['3214'])
    robot1.connect()
    robot2.connect()
    robot3.connect()

    for i in range(1,8):
        robot1.set_led(i, 1)
        robot2.set_led(i, 1)
        robot3.set_led(i, 1)

    log('Conection complete. CTRL+C to stop')
    log('Library version: ' + robot1.version)

except Exception, e:
    error(e)
    sys.exit(1)

n=3

roboid1 =1
roboid2 = 2
roboid3= 3

stoptag=27

epuckids=[roboid1,roboid2,roboid3]

tvecs = np.zeros((n, 3))
rvecs = np.zeros((n, 3))
angles = np.zeros((n, 3))

while (True):

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ###SELECT CORRECT DICTIONARY###

    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters_create()

    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    rvec, tvec, _objPoints = aruco.estimatePoseSingleMarkers(corners, markerLength, mtx, dist)

    # INITIAL STEP

    robot1.step()
    robot2.step()
    robot3.step()

    if np.all(ids != None):
        if (all(i in ids for i in epuckids)):
            for i in range(0, n):
                tvecs[i]= (tvec[np.where(ids == epuckids[i])][0])
                rvecs[i] = (rvec[np.where(ids == epuckids[i])][0])
                dst, jacobian = cv2.Rodrigues(rvecs[i])
                angles[i] = rotationMatrixToEulerAngles(dst)

                ### OWN CODE GOES HERE ###

                # TVECS=POSITION
                # ANGLES=ROTATION ANGLES






                # DRAW AXIS
                aruco.drawAxis(frame, mtx, dist, rvec[i], tvec[i], 0.1)

            if stoptag in ids:
                setvel([0, 0, 0] , [0, 0, 0])
                exit(6)

    frame = aruco.drawDetectedMarkers(frame, corners)
    font = cv2.FONT_HERSHEY_SIMPLEX

    ###### DRAW ID #####

    cv2.putText(frame, "Id: " + str(ids), (0, 64), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capt    ure
cap.release()
#out.release()
cv2.destroyAllWindows()



