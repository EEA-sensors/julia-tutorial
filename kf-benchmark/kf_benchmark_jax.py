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

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy import linalg

jax.config.update("jax_enable_x64", True)
#
# Load data
#
X = np.loadtxt('xdata.txt').T
Y_tmp = np.loadtxt('ydata.txt').T
Y = [np.array(Y_tmp[:, k]).T for k in range(Y_tmp.shape[1])]
Y = jnp.array(Y)
#
# Parameters
#
q = 1
dt = 0.1
s = 0.5

A = np.array([[1, 0, dt, 0],
              [0, 1, 0, dt],
              [0, 0, 1, 0],
              [0, 0, 0, 1]], dtype=float)

Q = q * np.array([[dt ** 3 / 3, 0, dt ** 2 / 2, 0],
                  [0, dt ** 3 / 3, 0, dt ** 2 / 2],
                  [dt ** 2 / 2, 0, dt, 0],
                  [0, dt ** 2 / 2, 0, dt]])

H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
R = s ** 2 * np.eye(2, dtype=float)
m0 = np.array([0, 0, 1, -1], dtype=float)
P0 = np.eye(4, dtype=float)

niter = 10000


#
# Kalman filter
#
@jax.jit
def kf_fn_jax():
    def body(carry, y):
        m, P = carry

        m = A @ m
        P = A @ P @ A.T + Q

        S = H @ P @ H.T + R
        K = linalg.solve(S, jnp.dot(H, P), assume_a="pos").T
        m += K @ (y - H @ m)
        P -= K @ H @ P

        return (m, P), (m, P)

    _, (ms, Ps) = jax.lax.scan(body, (m0, P0), Y)
    return ms, Ps


# compile run
kf_m, kf_P = kf_fn_jax()
kf_m.block_until_ready()

start_time = time.time()


for i in range(niter):
    kf_m, kf_P = kf_fn_jax()
    kf_m.block_until_ready()

print("Elapsed time %s seconds." % (time.time() - start_time))

print(f"{kf_m[-1]:}")
kf_m = np.array(kf_m)
kf_P = np.array(kf_P)
rmse_kf = 0
for k in range(len(kf_m)):
    rmse_kf += (kf_m[k][0] - X[0, k]) * (kf_m[k][0] - X[0, k]) + (kf_m[k][1] - X[1, k]) * (kf_m[k][1] - X[1, k])

rmse_kf = sqrt(rmse_kf / len(kf_m))
print(f"{rmse_kf:}")


#
# RTS smoother
#
@jax.jit
def rts_fn_jax_fn():
    kf_m_ = kf_m[::-1]
    kf_P_ = kf_P[::-1]
    def body(carry, inp):
        mf, Pf = inp
        prev_m, prev_P = carry

        mp = A @ mf
        Pp = A @ Pf @ A.T + Q

        K = linalg.solve(Pp, A @ Pf, assume_a="pos").T
        m = mf + K @ (prev_m - mp)
        P = Pf + K @ (prev_P - Pp) @ K.T

        return (m, P), (m, P)

    _, (ms, Ps) = jax.lax.scan(body, (kf_m_[0], kf_P_[0]), (kf_m_[1:], kf_P_[1:]))

    ms, Ps = ms[::-1], Ps[::-1]
    ms = jnp.concatenate((ms, kf_m_[0][None, :]), axis=0)
    Ps = jnp.concatenate((Ps, kf_P_[0][None, :, :]), axis=0)
    return ms, Ps

# compile run
rts_m, rts_P = rts_fn_jax_fn()
rts_m.block_until_ready()
start_time = time.time()

for i in range(niter):
    rts_m, rts_P = rts_fn_jax_fn()
    rts_m.block_until_ready()

print("Elapsed time %s seconds." % (time.time() - start_time))

rts_m = np.array(rts_m)
rts_P = np.array(rts_P)

print(f"{rts_m[0]:}")


rmse_rts = 0
for k in range(len(rts_m)):
    rmse_rts += (rts_m[k][0] - X[0, k]) * (rts_m[k][0] - X[0, k]) + (rts_m[k][1] - X[1, k]) * (rts_m[k][1] - X[1, k])

rmse_rts = sqrt(rmse_rts / len(rts_m))
print(f"{rmse_rts:}")
