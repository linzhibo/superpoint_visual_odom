import glob
import numpy as np 
import os
import time

import cv2
import torch

myjet = np.array([[0.        , 0.        , 0.5       ],
                  [0.        , 0.        , 0.99910873],
                  [0.        , 0.37843137, 1.        ],
                  [0.        , 0.83333333, 1.        ],
                  [0.30044276, 1.        , 0.66729918],
                  [0.66729918, 1.        , 0.30044276],
                  [1.        , 0.90123457, 0.        ],
                  [1.        , 0.48002905, 0.        ],
                  [0.99910873, 0.07334786, 0.        ],
                  [0.5       , 0.        , 0.        ]])

class SuperPointNet(torch.nn.Module):
	"""docstring for SuperPointNet"""
	def __init__(self, arg):
		super(SuperPointNet, self).__init__()
		self.relu = torch.nn.ReLU(inplace = True)
		self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
		c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
		# Shared Encoder.
	    self.conv1a = torch.nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
	    self.conv1b = torch.nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
	    self.conv2a = torch.nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
	    self.conv2b = torch.nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
	    self.conv3a = torch.nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
	    self.conv3b = torch.nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
	    self.conv4a = torch.nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
	    self.conv4b = torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
	    # Detector Head.
	    self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
	    self.convPb = torch.nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)
	    # Descriptor Head.
	    self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
	    self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)

	    def forward(self, x):
	    	# Shared Encoder.
		    x = self.relu(self.conv1a(x))
		    x = self.relu(self.conv1b(x))
		    x = self.pool(x)
		    x = self.relu(self.conv2a(x))
		    x = self.relu(self.conv2b(x))
		    x = self.pool(x)
		    x = self.relu(self.conv3a(x))
		    x = self.relu(self.conv3b(x))
		    x = self.pool(x)
		    x = self.relu(self.conv4a(x))
		    x = self.relu(self.conv4b(x))

		    cPa = self.relu(self.convPa(x))
		    detec = self.convPb(cPa)

		    cDa = self.relu(self.convDa(x))
		    desc = self.convDb(cDa)
		    dn = torch.norm(desc, p=2, dim=1)
		    desc = desc.div(torch.unsqueeze(dn, 1))
		    return detec, desc

class SuperPointFrontend(object):
	def __init__(self, weights_path, nms_dist, conf_thresh, nn_thresh, cuda=True):
		self.name = 'SuperPoint'
		self.cuda = cuda
		self.nms_dist = nms_dist
		self.conf_thresh = conf_thresh
		self.nn_thresh = nn_thresh
		self.cell = 8
		self.border_remove = 4

		self.net = SuperPointNet()
		self.net.load_state_dict(torch.load(weights_path))
		self.net = self.net.cuda()
		self.net.eval()

	def nms_fast(self, in_corners, H, W, dist_thresh):

		grid = np.zeros((H,W)).astype(int)
		inds = np.zeros((H,W)).astype(int)

		inds1 = np.argsort(-in_corners[2,:])
		corners = in_corners[:, inds1]
		rcorners = corners[:2,:].round().astype(int)

		if rcorners.shape[1] == 0:
			return np.zeros((3,0)).astype(int), np.zeros(0).astype(int)

		if rcorners.shape[1] == 1:
			return out, np.zeros((1)).astype(int)

		for i, rc in enumerate(rcorners.T):
			grid[rcorners[1,i], rcorners[0,i]] = 1
			inds[rcorners[1,i], rcorners[0,i]] = i

		pad = dist_thresh
		grid = np.pad(grid, ((pad, pad), (pad, pad)), mode = 'constant')

		count = 0
		for i, rc in enumerate(rcorners.T):
			pt = (rc[0] + pad, rc[1] + pad)
			if grid[pt[1], pt[0]] == 1:
				grid[pt[1]-pad:pt[1]+pad+1, pt[0]-pad:pt[0]+pad+1] = 0
				grid[pt[1], pt[0]] = -1
				count +=1
		keepy, keepx = np.where(grid==-1)
		keepy, keepx = keepy - pad, keepx - pad
		inds_keep = inds[keepy, keepx]
		out = corners[:,inds_keep]
		values = out[-1, :]
		inds2 = np.argsort(-values)
		out = out[:, inds2]
		out_inds = inds1[inds_keep[inds2]]
		return out, out_inds

	def run(self):
		assert img.ndim == 2, 'Image must be grayscale.'
	    assert img.dtype == np.float32, 'Image must be float32.'
	    H, W = img.shape[0], img.shape[1]
	    inp = img.copy()
	    inp = (inp.reshape(1, H, W))
	    inp = torch.from_numpy(inp)
	    inp = torch.autograd.Variable(inp).view(1, 1, H, W)
	    if self.cuda:
	      	inp = inp.cuda()
	    # Forward pass of network.
	    outs = self.net.forward(inp)
	    semi, coarse_desc = outs[0], outs[1]
	    # Convert pytorch -> numpy.
	    semi = semi.data.cpu().numpy().squeeze()
	    # --- Process points.
	    dense = np.exp(semi) # Softmax.
	    dense = dense / (np.sum(dense, axis=0)+.00001) # Should sum to 1.
	    # Remove dustbin.
	    nodust = dense[:-1, :, :]
	    # Reshape to get full resolution heatmap.
	    Hc = int(H / self.cell)
	    Wc = int(W / self.cell)
	    nodust = nodust.transpose(1, 2, 0)
	    heatmap = np.reshape(nodust, [Hc, Wc, self.cell, self.cell])
	    heatmap = np.transpose(heatmap, [0, 2, 1, 3])
	    heatmap = np.reshape(heatmap, [Hc*self.cell, Wc*self.cell])
	    xs, ys = np.where(heatmap >= self.conf_thresh) # Confidence threshold.
	    if len(xs) == 0:
	      	return np.zeros((3, 0)), None, None
	    pts = np.zeros((3, len(xs))) # Populate point data sized 3xN.
	    pts[0, :] = ys
	    pts[1, :] = xs
	    pts[2, :] = heatmap[xs, ys]
	    pts, _ = self.nms_fast(pts, H, W, dist_thresh=self.nms_dist) # Apply NMS.
	    inds = np.argsort(pts[2,:])
	    pts = pts[:,inds[::-1]] # Sort by confidence.
	    # Remove points along border.
	    bord = self.border_remove
	    toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (W-bord))
	    toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (H-bord))
	    toremove = np.logical_or(toremoveW, toremoveH)
	    pts = pts[:, ~toremove]
	    # --- Process descriptor.
	    D = coarse_desc.shape[1]
	    if pts.shape[1] == 0:
	      	desc = np.zeros((D, 0))
	    else:
	      # Interpolate into descriptor map using 2D point locations.
	      	samp_pts = torch.from_numpy(pts[:2, :].copy())
	      	samp_pts[0, :] = (samp_pts[0, :] / (float(W)/2.)) - 1.
	      	samp_pts[1, :] = (samp_pts[1, :] / (float(H)/2.)) - 1.
	      	samp_pts = samp_pts.transpose(0, 1).contiguous()
	      	samp_pts = samp_pts.view(1, 1, -1, 2)
	      	samp_pts = samp_pts.float()
	      	if self.cuda:
		        samp_pts = samp_pts.cuda()
	      	desc = torch.nn.functional.grid_sample(coarse_desc, samp_pts)
	      	desc = desc.data.cpu().numpy().reshape(D, -1)
	      	desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]
	    return pts, desc, heatmap

class PointTracker(object):
	def __init__(self, max_length, nn_thresh):
		if max_length < 2:
			raise ValueError('max_length must be greater than or equal to 2')
		self.maxl = max_length
		self.nn_thresh = nn_thresh
		self.all_pts = []
		for n in range(self.maxl):
			self.all_pts.append(np.zeros((2,0)))
		self.last_desc = None
		self.tracks = np.zeros((0, self.maxl+2))
		self.track_count = 0
		self.max_score = 9999

	def nn_match_two_way(self, desc1, desc2, nn_thresh):
		assert desc1.shape[0] == desc2.shape[0]
		if desc1.shape[1]==0 or desc2.shape[1]==0:
			return nn.zeros((3, 0))
		if nn_thresh < 0.0:
			raise ValueError('\'nn_thresh\' should be non-negative')

		dmat = np.dot(desc1.T, desc2)
		dmat = np.sqrt(2-2*np.clip(dmat, -1, 1))

		idx = np.argmin(dmat, axis=1)
		scores = dmat[np.arange(dmat.shape[0]), idx]

		keep = scores < nn_thresh

		idx2 = np.argmin(dmat, axis=0)
		keep_bi = np.arange(len(idx)) == idx2[idx]
		keep = np.logical_and(keep, keep_bi)
		idx = idx[keep]
		scores = scores[keep]

		m_idx1 = np.arange(desc1.shape[1])[keep]
		m_idx2 = idx

		matches = np.zeros((3, int(keep.sum())))
		matches[0, :] = m_idx1
		matches[1, :] = m_idx2
		matches[2, :] = scores
		return matches

	def get_offsets(self):
		offsets = []
		offsets.append(0)
		for i in range(len(self.all_pts) - 1):
			offsets.append(self.all_pts[i].shape[1])
		offsets = np.array(offsets)
		offsets = np.cumsum(offsets)
		return offsets
