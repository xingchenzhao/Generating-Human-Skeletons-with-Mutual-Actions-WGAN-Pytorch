#cython: language_level=3
import numpy as np
cimport numpy as np
import os
cimport cython

def read(fname, max_bodies=2):
    with open(fname, 'r') as f:
        num_frames = int(f.readline())
        keypoints = np.zeros((2, num_frames, 25, 2), dtype=np.float64)

        for t in range(num_frames):
            num_bodies = int(f.readline())

            for m in range(num_bodies):
                f.readline() # Body info, skip
                num_keypoints = int(f.readline())
                for k in range(num_keypoints): # Read joints
                    x, y = f.readline().split()[:2]
                    if m >= max_bodies:
                        continue

                    keypoints[m, t, k, 0] = x
                    keypoints[m, t, k, 1] = y
    return keypoints


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cpdef void ins_frames(double[:,:,:,::1] buf, double[:,:,:,::1] data, int diff):
    cdef int n0 = data.shape[1]
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
        buf[0, to_ins[i], 0, 0] = -10000001 # Marker

    recur = 0

    for i in range(buf.shape[1]):
        if buf[0, i, 0, 0] == -10000001:
            recur += 1
            continue

        for j in range(2):
            for k in range(25):
                for l in range(2):
                    v = data[j, count, k, l]
                    buf[j, i, k, l] = v # Copy
                    if recur != 0: # Calculate the mean
                        buf[j, i-1, k, l] = (v + data[j, count-1, k, l]) * 0.5

        if recur > 0: recur -= 1 # Reset

        count += 1
