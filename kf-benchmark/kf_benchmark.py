# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %
# % Modified from the following example file.
# %
# % Track car state with Kalman filter and Rauch-Tung-Striebel
# % smoother as in Examples 4.3 and 8.3 of the book
# %
# % Simo Sarkka (2013), Bayesian Filtering and Smoothing,
# % Cambridge University Press. 
# %
# % Last updated: $Date: 2013/08/26 12:58:41 $.
# %
# % This software is distributed under the GNU General Public 
# % Licence (version 2 or later); please refer to the file 
# % Licence.txt, included with the software, for details.
# %
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import time
from math import *
import numpy as np
from scipy import linalg


#
# Load data
#
X = np.loadtxt('xdata.txt').T
Y_tmp = np.loadtxt('ydata.txt').T
Y = [np.array(Y_tmp[:,k]).T for k in range(Y_tmp.shape[1])]


#
# Parameters
#
q = 1
dt = 0.1
s = 0.5

A = np.array([[1, 0, dt, 0],
     [0, 1, 0, dt],
     [0, 0, 1, 0],
     [0, 0, 0, 1]])
    
Q = q*np.array([[dt**3/3, 0, dt**2/2, 0],
       [0, dt**3/3, 0, dt**2/2],
       [dt**2/2, 0, dt, 0],
       [0 ,dt**2/2, 0, dt]])

H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
R = s**2*np.identity(2)
m0 = np.array([0, 0, 1, -1]).T
P0 = np.identity(4)

niter = 10000

#
# Kalman filter
#

kf_m = [np.zeros([P0.shape[0],1]) for k in range(len(Y))]
kf_P = [np.zeros([P0.shape[0],P0.shape[0]]) for k in range(len(Y))]

start_time = time.time()

for i in range(niter):
    m = m0
    P = P0
    for k in range(len(Y)):
        m = A.dot(m)
        P = A.dot(P).dot(A.T) + Q
        
        LL = linalg.cho_factor(H.dot(P).dot(H.T) + R)
        K = linalg.cho_solve(LL, H.dot(P.T)).T
        m += K.dot(Y[k]-H.dot(m))
        P -= K.dot(H).dot(P)
        
        kf_m[k] = m
        kf_P[k] = P

print("Elapsed time %s seconds." % (time.time() - start_time))
        
print(f"{m:}")

rmse_kf = 0
for k in range(len(kf_m)):
    rmse_kf += (kf_m[k][0] - X[0,k]) * (kf_m[k][0] - X[0,k]) + (kf_m[k][1] - X[1,k]) * (kf_m[k][1] - X[1,k])

rmse_kf = sqrt(rmse_kf / len(kf_m))
print(f"{rmse_kf:}")

#
# RTS smoother
#

rts_m = [np.zeros([P0.shape[0],1]) for k in range(len(Y))]
rts_P = [np.zeros([P0.shape[0],P0.shape[0]]) for k in range(len(Y))]

start_time = time.time()

for i in range(niter):
    ms = m
    Ps = P
    rts_m[-1] = ms
    rts_P[-1] = Ps
    for k in reversed(range(len(Y)-1)):
        mp = A.dot(kf_m[k])
        Pp = A.dot(kf_P[k]).dot(A.T) + Q
        
        LL = linalg.cho_factor(Pp)
        Ck = linalg.cho_solve(LL,A.dot(kf_P[k])).T
        ms = kf_m[k] + Ck.dot(ms - mp)
        Ps = kf_P[k] + Ck.dot(Ps - Pp).dot(Ck.T)
        
        rts_m[k] = ms
        rts_P[k] = Ps

print("Elapsed time %s seconds." % (time.time() - start_time))
        
print("{ms:}")

rmse_rts = 0
for k in range(len(rts_m)):
    rmse_rts += (rts_m[k][0] - X[0,k]) * (rts_m[k][0] - X[0,k]) + (rts_m[k][1] - X[1,k]) * (rts_m[k][1] - X[1,k])

rmse_rts = sqrt(rmse_rts / len(rts_m))
print(f"{rmse_rts:}")
