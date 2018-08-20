import numpy as np
import cv2
from iterative_closest_point import ICP_matching, SVD_motion_estimation

from spn import SuperPointFrontend, PointTracker

FIRST_FRAME = 0
SECOND_FRAME = 1
DEFAULT_FRAME = 2

class PinholeCamera(object):
	"""docstring for PinholeCamera"""
	def __init__(self, width, height, fx, fy, cx, cy,
		k1 = 0.0, k2=0.0, p1=0.0,p2=0.0,k3=0.0):
		self.width = width
		self.height = height
		self.fx = fx
		self.fy = fy
		self.cx = cx
		self.cy = cy
		self.distortion = (abs(k1) > 0.0000001)
		self.d = [k1, k2, p1, p2, k3]

class VisualOdometry():
	def __init__(self, cam):
		self.frame_stage = 0
		self.cam = cam
		self.new_frame = None
		self.last_frame = None
		self.cur_R = None
		self.cur_t = None
		self.px_ref = None
		self.px_cur = None
		self.focal = cam.fx
		self.pp = (cam.cx, cam.cy)
		self.trueX, self.trueY, self.trueZ = 0, 0, 0
		self.detector = SuperPointFrontend(weights_path = "superpoint_v1.pth",
										   nms_dist = 4,
										   conf_thresh = 0.015,
										   nn_thresh = 0.7,
										   cuda = True)
		self.tracker = PointTracker(max_length = 2,
									nn_thresh = self.detector.nn_thresh)
		# with open(annotations) as f:
		# 	self.annotations = f.readlines()

	def featureTracking(self):
		pts, desc, heatmap = self.detector.run(self.new_frame)
		self.tracker.update(pts, desc)
		tracks = self.tracker.get_tracks(min_length = 1)
		tracks[:, 1] /= float(self.detector.nn_thresh)
		kp1, kp2 = self.tracker.draw_tracks(tracks)
		return kp1, kp2

	# def getAbsoluteScale(self, frame_id):

	def processFirstFrame(self):
		self.px_ref, self.px_cur = self.featureTracking()
		self.frame_stage = SECOND_FRAME

	def processSecondFrame(self):
		self.px_ref, self.px_cur = self.featureTracking()

		E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref,
									   focal = self.focal,
									   pp = self.pp,
									   method = cv2.RANSAC, 
									   prob = 0.999,
									   threshold = 1.0)
		_, self.cur_R ,self.cur_t, mask = cv2.recoverPose(E, self.px_cur, self.px_ref,
														  focal = self.focal,
														  pp = self.pp)
		self.frame_stage = DEFAULT_FRAME
		self.px_ref = self.px_cur

	def processFrame(self, frame_id):
		self.px_ref, self.px_cur = self.featureTracking()

		E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref,
									   focal = self.focal,
									   pp = self.pp,
									   method = cv2.RANSAC, 
									   prob = 0.999,
									   threshold = 1)
		_, R, t, mask = cv2.recoverPose(E, self.px_cur, self.px_ref,
										focal = self.focal,
										pp = self.pp)
		# t_tmp = t + self.cur_t
		# distance = ((float(t_tmp[0]) - float(self.cur_t[0]))*(float(t_tmp[0]) - float(self.cur_t[0])) + 
		# 						 (float(t_tmp[1]) - float(self.cur_t[1]))*(float(t_tmp[1]) - float(self.cur_t[1]))  +
		# 						 (float(t_tmp[2]) - float(self.cur_t[2]))*(float(t_tmp[2]) - float(self.cur_t[2])) )
		# print (distance)
		# absolute_scale = pow(((t_tmp[0] - self.cur_t[0])*(t_tmp[0] - self.cur_t[0]) + 
		# 						 (t_tmp[1] - self.cur_t[1])*(t_tmp[1] - self.cur_t[1]) +
		# 						 (t_tmp[2] - self.cur_t[2])*(t_tmp[2] - self.cur_t[2])),0.5)
		# distance = float(t[0])*float(t[0]) + float(t[1])*float(t[1]) + float(t[2])*float(t[2])
		# print(t)
		# absolute_scale = pow(distance, 0.5)
		absolute_scale = 1
		# R, t = ICP_matching(self.px_ref,self.px_cur)
		# R, t = SVD_motion_estimation(self.px_cur,self.px_ref)
		self.cur_t = self.cur_t + absolute_scale* self.cur_R.dot(t)
		self.cur_R = R.dot(self.cur_R)
		self.px_ref = self.px_cur


	def update(self, img, frame_id):
		# assert(img.ndim == 2 and img.shape[0] == self.cam.height and img.shape[1] ==
  #              self.cam.width), "Frame: provided image has not the same size as the camera model or image is not grayscale"

		self.new_frame = img
		if(self.frame_stage == DEFAULT_FRAME):
			self.processFrame(frame_id)
		elif(self.frame_stage == SECOND_FRAME):
			self.processSecondFrame()
		elif(self.frame_stage == FIRST_FRAME):
			self.processFirstFrame()
		self.last_frame = self.new_frame
