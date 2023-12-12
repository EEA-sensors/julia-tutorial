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
import tensorflow as tf

mm = tf.linalg.matmul
mv = tf.linalg.matvec

#
# Load data
#
X = tf.convert_to_tensor(np.loadtxt('xdata.txt'), dtype=tf.float64)
Y = tf.convert_to_tensor(np.loadtxt('ydata.txt'), dtype=tf.float64)

#
# Parameters
#
q = 1
dt = 0.1
s = 0.5

A = tf.constant([[1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]], dtype=tf.float64)

Q = q * tf.constant([[dt ** 3 / 3, 0, dt ** 2 / 2, 0],
                    [0, dt ** 3 / 3, 0, dt ** 2 / 2],
                    [dt ** 2 / 2, 0, dt, 0],
                    [0, dt ** 2 / 2, 0, dt]], dtype=tf.float64)

H = tf.constant([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=tf.float64)
R = s ** 2 * tf.eye(2, dtype=tf.float64)
m0 = tf.constant([0, 0, 1, -1], dtype=tf.float64)
P0 = tf.eye(4, dtype=tf.float64)

niter = 10000

#
# Kalman filter
#

@tf.function
def kf(A, Q, H, R, Y, m0, P0):
    def body(carry, y):
        m, P = carry
        m = mv(A, m)
        P = A @ mm(P, A, transpose_b=True) + Q

        S = H @ mm(P, H, transpose_b=True) + R
        chol = tf.linalg.cholesky(S)
        Kt = tf.linalg.cholesky_solve(chol, H @ P)

        m = m + mv(Kt, y - mv(H, m), transpose_a=True)
        P = P - mm(Kt, S, transpose_a=True) @ Kt
        return m, P

    kf_m, kf_P = tf.scan(body, Y, (m0, P0))
    return kf_m, kf_P

def run_kf():
    kf_m, kf_P = kf(A, Q, H, R, Y, m0, P0)
    m = kf_m[-1].numpy()
    print(f"{m:}")

    start_time = time.time()

    for i in range(niter):
        kf_m, kf_P = kf(A, Q, H, R, Y, m0, P0)

    print("Elapsed time %s seconds." % (time.time() - start_time))

run_kf()

#
# Smoother
#

@tf.function
def rts(A, Q, kf_m, kf_P):
    def body(carry, inp):
        m, P = inp
        ms, Ps = carry

        mp = mv(A, m)
        Pp = A @ mm(P, A, transpose_b=True) + Q

        chol = tf.linalg.cholesky(Pp)
        Ct = tf.linalg.cholesky_solve(chol, A @ P)

        ms = m + mv(Ct, (ms - mp), transpose_a=True)
        Ps = P + mm(Ct, Ps - Pp, transpose_a=True) @ Ct
        return ms, Ps

    (sms, sPs) = tf.scan(body, (kf_m[:-1], kf_P[:-1]), (kf_m[-1], kf_P[-1]), reverse=True)
    rts_m = tf.concat([sms, tf.expand_dims(kf_m[-1], 0)], 0)
    rts_P = tf.concat([sPs, tf.expand_dims(kf_P[-1], 0)], 0)
    return rts_m, rts_P

def run_rts():
    kf_m, kf_P = kf(A, Q, H, R, Y, m0, P0)
    rts_m, rts_P = rts(A, Q, kf_m, kf_P)

    ms = rts_m[0].numpy()
    print(f"{ms:}")

    start_time = time.time()

    for i in range(niter):
        rts_m, rts_P = rts(A, Q, kf_m, kf_P)

    print("Elapsed time %s seconds." % (time.time() - start_time))

run_rts()
