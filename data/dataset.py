import numpy as np
import os
import torch.utils.data
import utils

class NTUSkeletonDataset(torch.utils.data.Dataset):
	def __init__(self, root_dir, frames=100, pinpoint=0, pin_body=None):
		"""
		root_dir: os.path or str
			Directory to the skeleton files
		frames: int
			The number of frames that all data will be aligned to
		pinpoint: int
			The index of the keypoint to pin at (0, 0)
		pin_body: int or None
			The index of the body. 
			If None, each body is normalized with respect to its pinpoint.
			Otherwise, all bodies are normalized with respect to the body
			indicated by `pin_body`.
		"""
		super().__init__()

		self.root_dir = root_dir
		self.files = os.listdir(root_dir)
		self.num_frames = frames
		self.pinpoint = pinpoint
		self.pin_body = pin_body

	def __len__(self):
		return len(self.files)

	def __getitem__(self, index):
		fname = self.files[index]

		# (# bodies, # keypoints, # frames, xy)
		f = utils.read(os.path.join(self.root_dir, fname))

		# Pin to one of the keypoints
		f = self._pin_skeleton(f)
		
		# Align the frames
		f = self._align_frames(f)
		assert f.shape[2] == self.num_frames, "wrong frames %d" % f.shape[2]

		return f

	def _pin_skeleton(self, data):
		if self.pin_body is None:
			pin_xyz = data[:, self.pinpoint, ...]
			data -= pin_xyz[:, None, ...]
		else:
			pin_xyz = data[self.pin_body, self.pinpoint, ...]
			data -= pin_xyz[None, None, ...]
		return data

	def _align_frames(self, data):
		num_frames0 = data.shape[2]
		diff = num_frames0 - self.num_frames

		if diff > 0: # Del
			to_del = np.linspace(0, num_frames0, num=diff, 
				endpoint=False, dtype=np.int32)
			return np.delete(data, to_del, axis=2)

		elif diff < 0: # Interpolate
			buf = np.zeros((2, 25, self.num_frames, 2), 
					dtype=np.float64)
			utils.ins_frames(buf, data, -diff)

			return buf

		else: # Keep as the original 
			return data
