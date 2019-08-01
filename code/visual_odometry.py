import numpy as np
import cv2
import time


STAGE_FIRST_FRAME = 0
STAGE_SECOND_FRAME = 1
STAGE_DEFAULT_FRAME = 2
kMinNumFeature = 20


class PinholeCamera:
	def __init__(self, width, height, fx, fy, cx, cy, k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0):
		self.width = width
		self.height = height
		self.fx = fx
		self.fy = fy
		self.cx = cx
		self.cy = cy
		self.distortion = (abs(k1) > 0.0000001)
		self.d = np.array([[k1, k2, p1, p2, k3]])
		self.mtx = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])


class DroneStates:
	def __init__(self):
		self.x = None
		self.y = None
		self.z = None
		self.phi = None
		self.theta = None
		self.psi = None
		self.u = None
		self.v = None
		self.w = None
		self.p = None
		self.q = None
		self.r = None
		self.heading_rate = None
		self.u_dot = None
		self.v_dot = None
		self.w_dot = None

		# Rotation from camera coordinates to drone coordinates
		self.R_cd = np.array([[0., 0., 1.],
							  [1., 0., 0.],
							  [0., 1., 0.]])

	def imu_data(self, data):
		self.p = data.gyro.x
		self.q = data.gyro.y
		self.r = data.gyro.z

	def set_depth(self, depth):
		self.z = depth.data

	def set_estimate(self, estimate):
		self.z = estimate.depth
		self.w = estimate.depth_rate
		self.psi = estimate.heading
		self.heading_rate = estimate.heading_rate


def featureTracking(image_prev, image_cur, px_prev):
	kp2, st, err = cv2.calcOpticalFlowPyrLK(image_prev, image_cur, px_prev, None, **lk_params)

	# select good points
	st = st.reshape(st.shape[0])
	kp1 = px_prev[st == 1]
	kp2 = kp2[st == 1]

	return kp1, kp2


class VisualOdometry:

	def __init__(self, cam):
		self.frame_stage = 0
		self.cam = cam
		self.new_frame = None
		self.last_frame = None
		self.R = np.zeros((3, 3))
		self.t = np.zeros((3, 1))
		self.cur_R = None
		self.cur_t = None
		self.prev_t = None
		self.px_prev = None
		self.px_cur = None
		self.kp_prev = None
		self.des_prev = None
		self.kp_cur = None
		self.des_cur = None
		self.focal = cam.fx
		self.pp = (cam.cx, cam.cy)
		self.trueX, self.trueY, self.trueZ = 0, 0, 0
		self.scale = 1.0
		self.detector = cv2.ORB_create()
		self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
		self.good = None
		self.matches = None

	def getAbsoluteScale(self):
		x, y, z = self.cur_t
		#x_prev, y_prev, z_prev = self.prev_t
		return np.sqrt(x**2 + y**2 + z**2)

	def processFirstFrame(self):
		self.kp_prev, self.des_prev = self.detector.detectAndCompute(self.new_frame, None)
		self.frame_stage = STAGE_SECOND_FRAME

	def processSecondFrame(self):
		self.kp_cur, self.des_cur = self.detector.detectAndCompute(self.new_frame, None)
		self.matches = self.bf.knnMatch(self.des_prev, self.des_cur, k=2)
		px_prev = []
		px_cur = []
		self.good = []
		for m,n in self.matches:
			if m.distance < 0.8 * n.distance:
				self.good.append(m)
				px_prev.append(tuple(self.kp_prev[m.queryIdx].pt))
				px_cur.append(tuple(self.kp_cur[m.trainIdx].pt))

		px_prev = np.asarray(px_prev)
		px_cur = np.asarray(px_cur)
		E, mask = cv2.findEssentialMat(px_cur, px_prev, focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.99, threshold=0.5)
		_, self.cur_R, self.cur_t, mask = cv2.recoverPose(E, px_cur, px_prev, focal=self.focal, pp=self.pp)
		self.frame_stage = STAGE_DEFAULT_FRAME
		self.des_prev = self.des_cur
		self.kp_prev = self.kp_cur

	def processFrame(self):
		self.matches = []
		self.good = []
		self.kp_cur, self.des_cur = self.detector.detectAndCompute(self.new_frame, None)
		self.matches = self.bf.knnMatch(self.des_prev, self.des_cur, k=2)
		px_prev = []
		px_cur = []
		for m,n in self.matches:
			if m.distance < 0.8 * n.distance:
				self.good.append(m)
				px_prev.append(tuple(self.kp_prev[m.queryIdx].pt))
				px_cur.append(tuple(self.kp_cur[m.trainIdx].pt))

		px_prev = np.asarray(px_prev)
		px_cur = np.asarray(px_cur)
		E, mask = cv2.findEssentialMat(px_cur, px_prev, focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.99, threshold=0.5)
		if E is None:
			return
		_, self.R, self.t, mask = cv2.recoverPose(E, px_cur, px_prev, focal=self.focal, pp=self.pp)

		self.cur_t = self.cur_t + self.scale * self.cur_R.dot(self.t)
		self.cur_R = self.R.dot(self.cur_R)

		self.des_prev = self.des_cur
		self.kp_prev = self.kp_cur

	def update(self, img):
		# assert(img.ndim==2 and img.shape[0]==self.cam.height and img.shape[1]==self.cam.width), "Frame: provided image has not the same size as the camera model or image is not grayscale"
		self.new_frame = img
		if self.frame_stage == STAGE_DEFAULT_FRAME:
			self.processFrame()
		elif self.frame_stage == STAGE_SECOND_FRAME:
			self.processSecondFrame()
		elif self.frame_stage == STAGE_FIRST_FRAME:
			self.processFirstFrame()
		self.last_frame = self.new_frame
