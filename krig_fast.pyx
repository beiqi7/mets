# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as np
from libc.math cimport sqrt, exp, fabs

def fast_variogram_spherical(double[:] h, double sill, double range_param, double nugget):
    """快速计算球形变异函数"""
    cdef int n = h.shape[0]
    cdef np.ndarray[double, ndim=1] result = np.zeros(n, dtype=np.float64)
    cdef int i
    cdef double h_scaled
    
    for i in range(n):
        if h[i] == 0:
            result[i] = 0
        else:
            h_scaled = h[i] / range_param
            if h_scaled >= 1.0:
                result[i] = sill + nugget
            else:
                result[i] = nugget + sill * (1.5 * h_scaled - 0.5 * h_scaled * h_scaled * h_scaled)
    
    return result

def fast_distance_matrix(double[:, :] points1, double[:, :] points2):
    """快速计算距离矩阵"""
    cdef int n1 = points1.shape[0]
    cdef int n2 = points2.shape[0]
    cdef np.ndarray[double, ndim=2] result = np.zeros((n1, n2), dtype=np.float64)
    cdef int i, j
    cdef double dx, dy
    
    for i in range(n1):
        for j in range(n2):
            dx = points1[i, 0] - points2[j, 0]
            dy = points1[i, 1] - points2[j, 1]
            result[i, j] = sqrt(dx*dx + dy*dy)
    
    return result 
