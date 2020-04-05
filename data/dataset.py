import numpy as np
import os
import torch.utils.data
from ntu_read_skeleton import read_xyz

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
		f = read_xyz(os.path.join(self.root_dir, fname))

		# Re-order as (# bodies, # keypoints, # frames, xy)
		f = f.astype(np.float32).transpose((3, 2, 1, 0))

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
			to_ins = np.linspace(1, num_frames0, num=-diff, 
				endpoint=False, dtype=np.int32)
			for i in range(to_ins.shape[0]):
				avg = (data[..., to_ins[i]-1, :] + data[..., to_ins[i], :]) / 2
				data = np.insert(data, to_ins[i], avg, axis=2)
				to_ins += 1 # Insert to the next position

			return data

		else: # Keep as the original 
			return data
