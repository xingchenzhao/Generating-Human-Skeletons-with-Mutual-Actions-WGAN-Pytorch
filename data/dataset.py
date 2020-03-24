import numpy as np
import os
import torch.utils.data
from ntu_read_skeleton import read_xyz

class NTUSkeletonDataset(torch.utils.data.Dataset):
	def __init__(self, root_dir, pinpoint=0, pin_body=None):
		"""
		root_dir: os.path or str
			Directory to the skeleton files
		pinpoint: int
			The index of the keypoint to pin at (0, 0, 0)
		pin_body: int or None
			The index of the body. 
			If None, each body is normalized with respect to its pinpoint.
			Otherwise, all bodies are normalized with respect to the body
			indicated by `pin_body`.
		"""
		super().__init__()

		self.root_dir = root_dir
		self.files = os.listdir(root_dir)
		self.pinpoint = pinpoint
		self.pin_body = pin_body

	def __len__(self):
		return len(self.files)

	def __getitem__(self, index):
		fname = self.files[index]
		f = read_xyz(os.path.join(self.root_dir, fname))

		# Re-order as (# bodies, # keypoints, # frames, xyz)
		f = f.astype(np.float32).transpose((3, 2, 1, 0))

		# Pin to one of the keypoints
		f = self._pin_skeleton(f)

		return f

	def _pin_skeleton(self, data):
		if self.pin_body is None:
			pin_xyz = data[:, self.pinpoint, ...]
			data -= pin_xyz[:, None, ...]
		else:
			pin_xyz = data[self.pin_body, self.pinpoint, ...]
			data -= pin_xyz[None, None, ...]
		return data
