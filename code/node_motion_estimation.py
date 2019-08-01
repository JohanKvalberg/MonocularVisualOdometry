#!/usr/bin/env python

import cv2
import numpy as np
import math
import rospy
import time
import csv

from visual_odometry import PinholeCamera, VisualOdometry, DroneStates
from VO_KF import VOKalmanFilter
from p2_drone.msg import StateEstimate, ImuData, Reference
from std_msgs.msg import Float32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import signal
import sys



def isRotationMatrix(R):
	# Checks if a matrix is a valid rotation matrix.
	Rt = np.transpose(R)
	shouldBeIdentity = np.dot(Rt, R)
	I = np.identity(3, dtype=R.dtype)
	n = np.linalg.norm(I - shouldBeIdentity)
	return n < 1e-6


def rotationMatrixToEulerAngles(R):
	# Calculates rotation matrix to euler angles
	# The result is the same as MATLAB except the order
	# of the euler angles ( x and z are swapped ).
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


#The controlles is not used due to unstable motion estimates
class controller:
	def __init__ (self, Kp, Ki, Kd):
		self.init_ref = None
		self.init_depth = None
		self.error = None
		self.Kp = Kp
		self.Ki = Ki
		self.Kd = Kd
		self.error_list = []
		self.time_list = []

	def bound(self, new_ref):
		if new_ref > 0.9:
			new_ref = 0.9
		elif new_ref < -0.9:
			new_ref = -0.9
		return new_ref

	def pid(self, ref, dt):
		if self.init_ref is None:
			self.init_ref = ref
		self.error = self.init_ref - ref
		if len(self.error_list) < 10:
			self.error_list.append(self.error)
			self.time_list.append(dt)
		else:
			del self.error_list[0]
			self.error_list.append(self.error)
			del self.time_list[0]
			self.time_list.append(dt)

		# PID controller
		# -- P --
		proportional = self.Kp*self.error

		# -- I --
		error_integral = 0.
		for i in range(len(self.error_list)):
			error_integral += self.error_list[i]*self.time_list[i]
		integral = self.Ki * error_integral

		# -- D --
		if len(self.error_list) < 10:
			derivative = 0.
		else:
			error_rate = (self.error_list[-1] - self.error_list[0])/sum(self.time_list)
			derivative = self.Kd * error_rate
		return self.bound(proportional + integral + derivative)


#Caluclation of the scale parameter based on the depth sensor
def depth_scale(control, depth, z, vo):
	if control.init_depth is None:
		control.init_depth = depth
		control.init_ref = control.init_depth - 0.2
	if (depth > (control.init_ref - 0.03)) and (depth < (control.init_ref + 0.03)):
		vo.scale = abs((depth - control.init_depth)/z)
		print("Scale found: %2f m/px" % vo.scale)
		vo.cur_t = np.array([[0.], [0.], [0.]])
		vo.cur_R = np.eye(3)
		# control.init_ref = control.init_depth


class image_converter:
	def __init__ (self):
		self.bridge = CvBridge()
		self.cv_image = None

	def callback(self, image):
		try:
			self.cv_image = self.bridge.imgmsg_to_cv2(image, "bgr8")
		except CvBridgeError as e:
			print(e)

def visual_odometry():
	# ---------------- Initial definitions -----------------------

	drone_states = DroneStates()
	ic = image_converter()
	#pose = Reference()

	rospy.Subscriber('sensor_depth/depth', Float32, drone_states.set_depth)
	rospy.Subscriber('sensor_imu_1/data', ImuData, drone_states.imu_data)
	rospy.Subscriber('camera/image_raw', Image, ic.callback)
	rospy.Subscriber('observer/state_estimate', StateEstimate, drone_states.set_estimate)

	rospy.init_node('visual_odometry', anonymous=True)

	#pub = rospy.Publisher('reference_generator/reference', Reference, queue_size=1)

	# Camera matrix is for image size 1920 x 1080
	mtx = np.array([[1.35445761E+03,	0.00000000E+00,	    8.91069717E+02],
			[0.00000000E+00,	1.37997405E+03,	    7.56192877E+02],
			[0.00000000E+00,	0.00000000E+00,	    1.00000000E+00]])

	dist = np.array([-0.2708139, 0.20052465, 2.08302E-02, 0.0002806, -0.10134601])

	cam = PinholeCamera(960, 540, mtx[0, 0]/2, mtx[1, 1]/2, mtx[0, 2]/2, mtx[1, 2]/2)

	# --------------------------------------------------------------

	vo = VisualOdometry(cam)
	kf = VOKalmanFilter()

	traj = np.zeros((600,650,3), np.uint8)

	init = True

	row = ['p_x', 'p_y', 'p_z', 'time', 'x_hat', 'y_hat', 'z_hat', 'a_x', 'a_y', 'a_z']
	f = open('VOestimates.csv', 'w+')
	writer = csv.writer(f)
	writer.writerow(row)
	x_hat = np.zeros((12, 1))
	drone_t = np.zeros((3,1))

	reference = Reference()

	# controller setup:
	surge_controller = 	controller(0.1, 0., 0.)
	sway_controller = 	controller(0.08, 0., 0.)
	yaw_controller = 	controller(0.009, 0., 0.005)
	depth_controller = 	controller(3.5, 0., 0.0)
	depth_controller.init_ref = 0.5

	rate = rospy.Rate(7.5)

	while not rospy.is_shutdown():
	# while i < len(depth_data.depth):
		t = time.time()
		#ret, frame = cap.read()
		#if not ret:
	#		continue

		frame = ic.cv_image
		#print(frame)
		frame = cv2.undistort(frame, mtx, dist)
		frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# run visual odometry
		vo.update(frame)

		if init:
			x, y, z, = 0., 0., 0.
			dt = time.time() - t

			init = False

		else:
			dt = time.time() - t
			cur_t = drone_states.R_cd.dot(vo.cur_t)
			x, y, z = cur_t[0, 0], cur_t[1, 0], cur_t[2, 0]

			roll,pitch,yaw = rotationMatrixToEulerAngles(vo.cur_R)

			keypoints = []
			for m in vo.good:
				keypoints.append(vo.kp_cur[m.trainIdx])
			frame = cv2.drawKeypoints(frame, keypoints, np.array([]), color=(0, 255, 0), flags=0)


		# Kalman filtering of the estimates 
		#if isinstance(vo.scale, np.float64):
		#	dt = time.time() - t
		#	drone_t = drone_states.R_cd.dot(vo.t)
		#	u = np.array([[x], [y], [drone_states.z], [drone_states.p], [drone_states.q], [drone_states.r]])
		#	kf.update(u, dt)
		#	x_hat = kf.x_hat * dt

		# write estimates to file for plotting
		row = [x, y, z, roll, pitch, yaw]
		writer.writerow(row)

		draw_x, draw_y = int(y) * (1) + 290, int(x) * (-1) + 290
		cv2.circle(traj, (draw_x, draw_y), 1, (0, 255, 0), 1)
		cv2.rectangle(traj, (10, 20), (650, 60), (0, 0, 0), -1)
		text = "Coordinates: x=%2fm y=%2fm z=%2fm fps: %f" % (x_filtered, y_filtered, z_filtered, 1 / dt)
		cv2.putText(traj, text, (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)

		cv2.imshow('Trajectory', traj)

		cv2.imshow('Video', frame)

		ch = cv2.waitKey(1)
		if ch & 0xFF == ord('q'):
			f.close()
			break

		# -------------CONTROLLERS-------------------
		if vo.scale == 1.0:
			reference.heave = depth_controller.pid(drone_states.z, dt)
			depth_scale(depth_controller, drone_states.z, z, vo)
			reference.surge = 0.
			reference.yaw = -0.05*reference.heave
			reference.sway = -0.3 * reference.heave

		if isinstance(vo.scale, np.float64):
			reference.surge = surge_controller.pid(x_filtered, dt)
			reference.sway = sway_controller.pid(y_filtered, dt)
			if -0.1 < reference.sway < 0:
				reference.sway = -0.1
			reference.yaw = yaw_controller.pid(drone_states.psi, dt)
			reference.heave = depth_controller.pid(drone_states.z, dt)
	

		reference.depth = 0.
		reference.depth_rate = 0.
		reference.heading = 0.
		reference.heading_rate = 0.
	
		rate.sleep()

	reference.surge = 0.
	reference.sway = 0.
	reference.yaw = 0.
	reference.depth = 0.
	reference.depth_rate = 0.
	reference.heading = 0.
	reference.heading_rate = 0.
	pub.publish(reference)

if __name__ == '__main__':
	visual_odometry()
