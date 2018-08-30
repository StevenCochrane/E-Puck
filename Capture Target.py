from ePuck import ePuck
import sys
import numpy as np
import cv2
import cv2.aruco as aruco
import math
import matplotlib as mat
import matplotlib.pyplot as plt
import time

def satv(v):

    vmax=0.13
    vmin=-0.13

    for i in range(0,len(v)):
        v[i] = min(vmax, max(vmin, v[i]))

    return v

def satw(w):

    wmax = 4.5
    wmin = -4.5

    for i in range(0, len(w)):
        w[i] = min(wmax, max(wmin, w[i]))

    return w

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


def setvel(v, w):
    # assert v < 0.12

    L = 0.053  # Axel Width
    R = 0.0205  # Wheel Radius

    vr1 = (2 * v[0] + w[0] * L) / (2 * R)
    vl1 = (2 * v[0] - w[0] * L) / (2 * R)
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

    print(ls1)
    print(rs1)

    robot1.set_motors_speed(ls1, rs1)
    robot1.step()
    robot2.set_motors_speed(ls2, rs2)
    robot2.step()
    robot3.set_motors_speed(ls3, rs3)
    robot3.step()


epucks = {
    '3303': '10:00:e8:c5:64:37',
    '3214': '10:00:e8:c5:64:56',
    '3281': '10:00:e8:c5:61:82',
    '3276': '10:00:e8:c5:61:43',
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

# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi',fourcc, 30.0, (640,480))

cv_file = cv2.FileStorage("calib.yaml", cv2.FILE_STORAGE_READ)

# note we also have to specify the type to retrieve other wise we only get a
# FileNode object back instead of a matrix

mtx = cv_file.getNode("camera_matrix").mat()
dist = cv_file.getNode("dist_coeff").mat()

markerLength = 0.08

print('Connecting with the ePuck')

try:

    # robot1 = ePuck(epucks['3109'])
    # robot2 = ePuck(epucks['3276'])
    # robot3 = ePuck(epucks['3214'])


    robot1 = ePuck(epucks['3214'])
    robot2 = ePuck(epucks['3276'])
    robot3 = ePuck(epucks['3281'])
    # robot4 = ePuck(epucks['3109'])


    # Second, connect to it

    robot1.connect()
    robot2.connect()
    robot3.connect()

    for i in range(1, 8):
        robot1.set_led(i, 1)
        robot2.set_led(i, 1)
        robot3.set_led(i, 1)
    log('Conection complete. CTRL+C to stop')
    log('Library version: ' + robot1.version)

except Exception, e:
    error(e)
    sys.exit(1)

roboid1 = 3
roboid2 = 2
roboid3 = 1

stoptag = 27

epuckids = [roboid1, roboid2, roboid3]

n = 3

tvecs = np.zeros((n, 3))
rvecs = np.zeros((n, 3))
angles = np.zeros((n, 3))

p=1500

w = np.zeros((n, 1))
v = np.zeros((n, 1))
x = np.zeros((p, n))
y = np.zeros((p, n))
t = np.zeros((p, 1))
vrecord = np.zeros((p, n))
wrecord = np.zeros((p, n))
told=time.time()

count=0

distance = 0.3

kv = 0.5

wv = 1

vmax = 0.13
wmax = 1

phi = -math.atan(distance*wv*wmax/(kv*vmax))

theta = np.zeros((n, 1))

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

    if (count<p):
        if np.all(ids != None):
            if (all(i in ids for i in epuckids)):
                for i in range(0, n):
                    tvecs[i] = (tvec[np.where(ids == epuckids[i])][0])
                    rvecs[i] = (rvec[np.where(ids == epuckids[i])][0])
                    dst, jacobian = cv2.Rodrigues(rvecs[i])
                    angles[i] = rotationMatrixToEulerAngles(dst)
                    g=1
                    for j in range(0,n-1):
                        if j!=i:
                            a = ((math.atan2(tvecs[j][1] - tvecs[i][1],
                                        tvecs[j][0] - tvecs[i][0]) + 2 * math.pi) % (2 * math.pi) - (
                                    angles[i][2] + 2 * math.pi) % (2 * math.pi) +2*math.pi)%(2*math.pi)
                            g=g-math.cos(a)

                    b = ((math.atan2(tvecs[n - 1][1] - tvecs[i][1],
                                tvecs[n - 1][0] - tvecs[i][0]) + 2 * math.pi) % (2 * math.pi) - (
                            angles[i][2] + 2 * math.pi) % (2 * math.pi) +2*math.pi)%(2*math.pi)

                    aruco.drawAxis(frame, mtx, dist, rvec[i], tvec[i], 0.1)  # Draw Axis

                    # a = ((math.atan2(tvecs[i % (n - 1) + 1][1] - tvecs[i][1],
                    #             tvecs[i % (n - 1) + 1][0] - tvecs[i][0]) + 2 * math.pi) % (2 * math.pi) - (
                    #         angles[i][2] + 2 * math.pi) % (2 * math.pi) +2*math.pi)%(2*math.pi)
                    #
                    # b = ((math.atan2(tvecs[n - 1][1] - tvecs[i][1],
                    #             tvecs[n - 1][0] - tvecs[i][0]) + 2 * math.pi) % (2 * math.pi) - (
                    #         angles[i][2] + 2 * math.pi) % (2 * math.pi) +2*math.pi)%(2*math.pi)
                    #
                    # aruco.drawAxis(frame, mtx, dist, rvec[i], tvec[i], 0.1)  # Draw Axis

                    # g=(2*math.pi*a-(a*a))/(math.pi*math.pi)

                    v[i] = g*kv * vmax * math.cos(b + phi)

                    # if (v[i]<0.01):
                    #     v[i]=0.01

                    w[i] = -g*wv * wmax * math.sin(b + phi)

                    # v[i] = kv * vmax * math.cos(b + phi)
                    # if (v[i] < 0.01):
                    #     v[i] = 0.01
                    # w[i] = -wv * wmax * math.sin(b + phi)

                    x[count][i]=tvecs[i][0]
                    y[count][i]=tvecs[i][1]

                v[2]=0
                w[2]=0
                v=satv(v)
                w=satw(w)
                vrecord[count] = v.transpose()
                wrecord[count] = w.transpose()
                setvel(v, w)
                t[count] = time.time() - told

                count+=1

            if stoptag in ids:
                setvel([0, 0, 0], [0, 0, 0])
                plt.plot(x,y)
                plt.show()
                exit(6)
    else:
        setvel([0, 0, 0], [0, 0, 0])
        np.savetxt('nc2/x_nc2.csv', x, delimiter=",")
        np.savetxt('nc2/y_nc2.csv', y, delimiter=",")
        np.savetxt('nc2/time_nc2.csv', t, delimiter=",")
        np.savetxt('nc2/v_nc2.csv', vrecord, delimiter=",")
        np.savetxt('nc2/w_nc2.csv', wrecord, delimiter=",")
        plt.plot(x, y)
        plt.show()
        exit(6)

    frame = aruco.drawDetectedMarkers(frame, corners)
    font = cv2.FONT_HERSHEY_SIMPLEX  # font for displaying text (below)
    ###### DRAW ID #####
    cv2.putText(frame, "Id: " + str(ids), (0, 64), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
