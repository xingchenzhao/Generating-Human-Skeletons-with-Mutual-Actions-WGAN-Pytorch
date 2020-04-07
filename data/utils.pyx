#cython: language_level=3
import numpy as np
cimport numpy as np
import os
cimport cython

def read(fname, max_bodies=2):
    with open(fname, 'r') as f:
        num_frames = int(f.readline())
        keypoints = np.zeros((2, 25, num_frames, 3), dtype=np.float64)

        for t in range(num_frames):
            num_bodies = int(f.readline())

            for m in range(num_bodies):
                f.readline() # Body info, skip
                num_keypoints = int(f.readline())
                for k in range(num_keypoints): # Read joints
                    x, y, z = f.readline().split()[:3]
                    if m >= max_bodies:
                        continue

                    keypoints[m, k, t, 0] = x
                    keypoints[m, k, t, 1] = y
                    keypoints[m, k, t, 2] = z
    return keypoints


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cpdef void ins_frames(double[:,:,:,::1] buf, double[:,:,:,::1] data, int diff):
    cdef int n0 = data.shape[2]
    cdef int i = 0
    cdef int j = 0
    cdef int k = 0
    cdef int l = 0
    cdef double v = 0
    cdef int count = 0

    indices = np.linspace(1, n0, num=diff, endpoint=False, dtype=np.int32) \
              + np.arange(diff, dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] to_ins = indices
    
    for i in range(to_ins.shape[0]):
        buf[0, 0, to_ins[i], 0] = -10000001 # Marker

    recur = False

    for i in range(buf.shape[2]):
        if buf[0, 0, i, 0] == -10000001:
            recur = True
            continue

        for j in range(2):
            for k in range(25):
                for l in range(3):
                    v = data[j, k, count, l]
                    buf[j, k, i, l] = v # Copy
                    if recur: # Calculate the mean
                        buf[j, k, i-1, l] = (v + data[j, k, count-1, l]) * 0.5
                        recur = False # Reset

        count += 1
