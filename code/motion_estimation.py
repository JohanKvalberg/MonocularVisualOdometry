#!/usr/bin/env python

import cv2
import numpy as np
import rospy
import time
import csv
import math

from visual_odometry import PinholeCamera, VisualOdometry, DroneStates
#from p2_drone.msg import StateEstimate, ImuData, Reference
#from std_msgs.msg import Float32
#from sensor_msgs.msg import Image
#from cv_bridge import CvBridge, CvBridgeError

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
	Rt = np.transpose(R)
	shouldBeIdentity = np.dot(Rt, R)
	I = np.identity(3, dtype=R.dtype)
	n = np.linalg.norm(I - shouldBeIdentity)
	return n < 1e-6


# Calculates rotation matrix to euler angles
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

def visual_odometry():
	# ---------------- Initial definitions -----------------------

	drone_states = DroneStates()
	#ic = image_converter()


	#rospy.Subscriber('camera/image_raw', Image, ic.callback)
	#rospy.init_node('visual_odometry', anonymous=True)

	# --- Calibration data ---
	# -- Blueye --
	# Camera matrix: 
	 mtx = np.array([[1.35445761E+03,	0.00000000E+00,	    8.91069717E+02],
	   				[0.00000000E+00,	1.37997405E+03,	    7.56192877E+02],
	   				[0.00000000E+00,	0.00000000E+00,	    1.00000000E+00]])
	# Distortion coefficients 
	 dist = np.array([-0.2708139, 0.20052465, 2.08302E-02, 0.0002806, -0.10134601])


	# -- Blueye in-air --
	# Camera matrix:
	#mtx = np.array([[978.36617202, 0., 985.08473535],
	#				[0., 975.69987506, 541.52130078],
	#				[0., 0., 1.]])

	# Distortion coefficients 
	#dist = np.array([-0.37370139, 0.26899755, -0.00120655, -0.00185788, -0.1411856])

	cam = PinholeCamera(1920, 1080, mtx[0, 0], mtx[1, 1], mtx[0, 2], mtx[1, 2])

	# --------------------------------------------------------------

	vo = VisualOdometry(cam)

	traj = np.zeros((600,650,3), np.uint8)

	init = True

	# controller setup:
	# surge_controller = 	controller(0.1, 0., 0.)
	# sway_controller = 	controller(0.08, 0., 0.)
	# yaw_controller = 	controller(0.009, 0., 0.005)
	# depth_controller = 	controller(3.5, 0., 0.0)
	# depth_controller.init_ref = 0.5

	cap = cv2.VideoCapture("pool.MP4")

	row = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']
	f = open('VO_results.csv', 'w+')
	writer = csv.writer(f)
	writer.writerow(row)

	roll = 0
	pitch = 0
	yaw = 0

	while not rospy.is_shutdown():
		t = time.time()
		ret, frame = cap.read()
		if not ret:
			continue

		#frame = ic.cv_image

		frame = cv2.undistort(frame, mtx, dist)
		#frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# run visual odometry
		vo.update(frame)

		if init:
			x, y, z, = 0., 0., 0.
			dt = time.time() - t

			init = False

		else:
			vo.scale = 0.088152 	#Median from scale tests
			dt = time.time() - t
			cur_t = drone_states.R_cd.dot(vo.cur_t)
			roll, pitch, yaw = rotationMatrixToEulerAngles(vo.cur_R)
			x, y, z = cur_t[0, 0], cur_t[1, 0], cur_t[2, 0]

			keypoints = []
			for m in vo.good:
				keypoints.append(vo.kp_cur[m.trainIdx])
			frame = cv2.drawKeypoints(frame, keypoints, np.array([]), color=(0, 255, 0), flags=0)

		draw_x, draw_y = int(x) + 290, int(y) + 290
		cv2.circle(traj, (draw_x, draw_y), 1, (0, 255, 0), 1)
		cv2.rectangle(traj, (10, 20), (650, 60), (0, 0, 0), -1)
		text = "Coordinates: x=%2fm y=%2fm z=%2fm fps: %f" % (x, y, z, 1 / dt)
		cv2.putText(traj, text, (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)

		row = [x, y, z, roll, pitch, yaw]
		writer.writerow(row)

		cv2.imshow('Trajectory', traj)

		frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
		cv2.imshow('Video', frame)

		ch = cv2.waitKey(1)
		if ch & 0xFF == ord('q'):
			f.close()
			break


if __name__ == '__main__':
	visual_odometry()
