from ePuck import ePuck
import sys
import numpy as np
import cv2
import cv2.aruco as aruco
import math
import matplotlib.pyplot as plt
import time

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

def satv(v):

    vmax=0.13
    vmin=-0.13

    for i in range(0,len(v)):
        v[i] = min(vmax, max(vmin, v[i]))

    return v

def satw(w):

    wmax = 4.85
    wmin = -4.85

    for i in range(0, len(w)):
        w[i] = min(wmax, max(wmin, w[i]))

    return w


def setvel(v, w):
   # assert v < 0.12

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

    vr4 = (2 * v[3] + w[3] * L) / (2 * R)
    vl4 = (2 * v[3] - w[3] * L) / (2 * R)
    rs4 = vr4 / 0.00628
    ls4 = vl4 / 0.00628

    robot1.set_motors_speed(ls1, rs1)
    robot1.step()
    robot2.set_motors_speed(ls2, rs2)
    robot2.step()
    robot3.set_motors_speed(ls3, rs3)
    robot3.step()
    robot4.set_motors_speed(ls4, rs4)
    robot4.step()

epucks = {
    '3303': '10:00:e8:c5:64:37',
    '3214': '10:00:e8:c5:64:56',
    '3281': '10:00:e8:c5:61:82',
    '3276' : '10:00:e8:c5:61:43',
    '3109': '10:00:e8:ad:78:1d'
}


def log(text):
    """	Show @text in standart output with colors """

    blue = '\033[1;34m'
    off = '\033[1;m'

    print(''.join((blue, '[Log] ', off, str(text))))


def error(text):
    red = '\033[1;31m'
    off = '\033[1;m'

    print(''.join((red, '[Error] ', off, str(text))))


cap = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output2.avi',fourcc, 30.0, (640,480))

cv_file = cv2.FileStorage("calib.yaml", cv2.FILE_STORAGE_READ)

# note we also have to specify the type to retrieve other wise we only get a
# FileNode object back instead of a matrix

mtx = cv_file.getNode("camera_matrix").mat()
dist = cv_file.getNode("dist_coeff").mat()

markerLength = 0.15

global_speed = 180
fs_speed = 0.6
threshold = 1000

print('Connecting with the ePuck')

try:

    robot1 = ePuck(epucks['3281'])
    robot2 = ePuck(epucks['3276'])
    robot3 = ePuck(epucks['3214'])
    robot4 = ePuck(epucks['3303'])

    # Second, connect to it

    robot1.connect()
    robot2.connect()
    robot3.connect()
    robot4.connect()

    for i in range(1, 8):
        robot1.set_led(i, 1)
        robot2.set_led(i, 1)
        robot3.set_led(i, 1)
        robot4.set_led(i, 1)

    log('Conection complete. CTRL+C to stop')
    log('Library version: ' + robot1.version)

except Exception, e:
    error(e)
    sys.exit(1)

roboid1 = 1
roboid2 = 2
roboid3 = 3
robotid4= 6

stoptag = 27

epucks = [roboid1, roboid2, roboid3, robotid4]

n = 4

p=1000
count=0
told=time.time()
tvecs = np.zeros((n, 3))
rvecs = np.zeros((n, 3))
angles = np.zeros((n, 3))
bearing = np.zeros((n, n))

w = np.zeros((n, 1))
v = np.zeros((n, 1))
theta = np.zeros((4, 1))

x = np.zeros((p, n))
y = np.zeros((p, n))
t = np.zeros((p, 1))
vrecord = np.zeros((p, n))
wrecord = np.zeros((p, n))

thetastar = [[0, 0, math.pi / 4, math.pi / 2],
            [ math.pi, 0, math.pi / 2, 3 * math.pi / 4],
            [ 5 * math.pi / 4, 3 * math.pi / 2, 0, math.pi],
            [3 * math.pi / 2, 7 * math.pi / 4, 0, 0]]

# thetastar = [[0, 0, math.pi / 4, math.pi / 2],
#             [ math.pi, 0, math.pi / 2, 3 * math.pi / 4],
#             [0,0, 0, math.pi/3],
#             [3 * math.pi / 2, 7 * math.pi / 4, 0, 0]]

# thetastar = [[0, 0, 0, 0],
#              [math.pi, 0, 0, 0],
#              [math.pi, math.pi, 0, 0],
#              [math.pi, math.pi, math.pi, 0]]

while (True):

    ret, frame = cap.read()
    # operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters_create()

    # lists of ids and the corners beloning to each id
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    rvec, tvec, _objPoints = aruco.estimatePoseSingleMarkers(corners, markerLength, mtx, dist)  # For a single marker

    robot1.step()
    robot2.step()
    robot3.step()
    robot4.step()

    if(count<p):
        if np.all(ids != None):
            if (all(i in ids for i in epucks)):
                for i in range(0, n):
                    tvecs[i] = (tvec[np.where(ids == epucks[i])][0])
                    rvecs[i] = (rvec[np.where(ids == epucks[i])][0])
                    dst, jacobian = cv2.Rodrigues(rvecs[i])
                    angles[i] = rotationMatrixToEulerAngles(dst)
                    for j in range(0, n):
                        bearing[i][j] = (math.atan2(tvecs[j][1] - tvecs[i][1],
                                                    tvecs[j][0] - tvecs[i][0]) + 2 * math.pi) % (2 * math.pi)

                    theta[i] = (angles[i][2] + 2 * math.pi) % (2 * math.pi)
                    aruco.drawAxis(frame, mtx, dist, rvec[i], tvec[i], 0.1)  # Draw Axis

                    sum=np.array([0,0])
                    for j in range(0, n):
                        if (j != i):
                            g = np.array([math.cos(bearing[i][j]), math.sin(bearing[i][j])])
                            sum = sum + (np.eye(2) - np.outer(g, g)).dot(np.array([math.cos(thetastar[i][j]),
                                                                             math.sin(thetastar[i][j])]))
                    x[count][i] = tvecs[i][0]
                    y[count][i] = tvecs[i][1]

                    v[i] = (1) * np.array([-math.cos(theta[i]), -math.sin(theta[i])]).dot(sum)
                    w[i] = (1) * np.array([math.sin(theta[i]), -math.cos(theta[i])]).dot(sum)
                    v[0]=0
                    w[0]=0
                    v[1]=0
                    w[1]=0

                v=satv(v)
                w=satw(w)
                vrecord[count] = v.transpose()
                wrecord[count] = w.transpose()
                t[count] = time.time() - told
                print(vrecord[count])
                setvel(v, w)
                count=count+1

                if stoptag in ids:
                    setvel([0, 0, 0,0], [0, 0, 0,0])
                    plt.plot(x, y)
                    plt.show()
                    exit(6)
    else:
        setvel([0, 0, 0,0], [0, 0, 0,0])
        np.savetxt('shape/x_shape.csv', x, delimiter=",")
        np.savetxt('shape/y_shape.csv', y, delimiter=",")
        np.savetxt('shape/time_shape.csv', t, delimiter=",")
        np.savetxt('shape/v_shape.csv', vrecord, delimiter=",")
        np.savetxt('shape/w_shape.csv', wrecord, delimiter=",")
        plt.plot(x, y)
        plt.show()
        exit(6)

    frame = aruco.drawDetectedMarkers(frame, corners)
    font = cv2.FONT_HERSHEY_SIMPLEX  # font for displaying text (below)

    ###### DRAW ID #####
    # cv2.putText(frame, "Id: " + str(ids), (0, 64), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    out.write(frame)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

