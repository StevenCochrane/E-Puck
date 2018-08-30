import numpy as np
import cv2
import cv2.aruco as aruco
import yaml
import time
import math

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).

def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))

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

cap = cv2.VideoCapture(0)
cv_file = cv2.FileStorage("calib.yaml", cv2.FILE_STORAGE_READ)

#note we also have to specify the type to retrieve other wise we only get a
# FileNode object back instead of a matrix
mtx = cv_file.getNode("camera_matrix").mat()
dist = cv_file.getNode("dist_coeff").mat()
markerLength=0.1


while(True):

    # Capture frame-by-frame
    ret, frame = cap.read()
    #print(frame.shape) #480x640
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
    parameters =  aruco.DetectorParameters_create()

    #lists of ids and the corners beloning to each id
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    rvecs, tvecs, _objPoints = aruco.estimatePoseSingleMarkers(corners, markerLength, mtx, dist)  # For a single marker

    if np.all(ids != None):
        for i in ids:
            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, markerLength, mtx, dist)  # Estimate pose of each marker and return the values rvet and tvec---different from camera coefficients
            #rvec-tvec).any() # get rid of that nasty numpy value array error
            pos=np.array(tvec[0][0])
            angle=np.array(rvec[0][0])
            dst, jacobian = cv2.Rodrigues(angle)
            t=rotationMatrixToEulerAngles(dst)
            print(t)

            #print('x=', tvec[0][0][0]) # id / 0 / entry

    #It's working.
    # my problem was that the cellphone put black all around it. The alrogithm
    # depends very much upon finding rectangular black blobs

    gray = aruco.drawDetectedMarkers(gray, corners)

    #print(rejectedImgPoints)
    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


