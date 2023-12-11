
#=
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Modified from the following example file.
%
% Track car state with Kalman filter and Rauch-Tung-Striebel
% smoother as in Examples 4.3 and 8.3 of the book
%
% Simo Sarkka (2013), Bayesian Filtering and Smoothing,
% Cambridge University Press. 
%
% Last updated: $Date: 2013/08/26 12:58:41 $.
%
% This software is distributed under the GNU General Public 
% Licence (version 2 or later); please refer to the file 
% Licence.txt, included with the software, for details.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
=#

using DelimitedFiles
using LinearAlgebra
using StaticArrays

#
# Read data
#

X = readdlm("xdata.txt")';
X = [X[:,k] for k in 1:size(X,2)]
Y = readdlm("ydata.txt")';
Y = [Y[:,k] for k in 1:size(Y,2)]

#
# Set the parameters
#
q = 1.0;
dt = 0.1;
s = 0.5;
A = [1.0 0 dt 0;
    0 1.0 0 dt;
    0 0 1.0 0;
    0 0 0 1.0];
Q = q*[dt^3/3 0 dt^2/2 0;
        0 dt^3/3 0 dt^2/2;
        dt^2/2 0 dt 0;
        0 dt^2/2 0 dt];

H = [1.0 0 0 0;
        0 1.0 0 0];
R = s^2*I(2);
m0 = [0;0;1.0;-1.0];
P0 = I(4);

niter = 10000
println("niter = ", niter)

#
# Kalman filter
#
function kf!(kf_m, kf_P, Y, m0, P0, A, Q, H, R, niter)
    m = m0;
    P = P0;

    for i in 1:niter
        m = m0;
        P = P0;
        for k in 1:length(Y)
            m = A*m;
            P = A*P*A' + Q;

            S = H*P*H' + R;
            K = P*H'/S;
            m = m + K*(Y[k] - H*m);
            P = P - K*S*K';

            kf_m[k] = m;
            kf_P[k] = P;
        end
    end
end

#
# Run KF
#
function run_kf()
    kf_m = [zeros(size(m0)) for i in 1:length(Y)];
    kf_P = [zeros(size(P0)) for i in 1:length(Y)];

    kf!(kf_m, kf_P, Y, m0, P0, A, Q, H, R, 1);  # warmup

    @time kf!(kf_m, kf_P, Y, m0, P0, A, Q, H, R, niter);

    m = kf_m[end]
    println("m = ", m)

    rmse_kf = 0.0;
    for k in 1:length(kf_m)
        rmse_kf = rmse_kf + sum((kf_m[k][1:2,1] - X[k][1:2,1]).^2)
    end
    rmse_kf = sqrt(rmse_kf / length(kf_m))

    println("rmse_kf = ", rmse_kf)
end

run_kf();

#
# RTS smoother
#
function rts!(rts_m, rts_P, kf_m, kf_P, A, Q, niter)
    m = kf_m[end];
    P = kf_P[end];

    for i in 1:niter
        ms = m;
        Ps = P;    
        rts_m[end] = ms;
        rts_P[end] = Ps;
        for k in length(kf_m)-1:-1:1
            mp = A*kf_m[k];
            Pp = A*kf_P[k]*A'+Q;
            Ck = kf_P[k]*A'/Pp; 
            ms = kf_m[k] + Ck*(ms - mp);
            Ps = kf_P[k] + Ck*(Ps - Pp)*Ck';
            rts_m[k] = ms;
            rts_P[k] = Ps;
        end
    end
end

function run_rts()
    kf_m = [zeros(size(m0)) for i in 1:length(Y)];
    kf_P = [zeros(size(P0)) for i in 1:length(Y)];

    kf!(kf_m, kf_P, Y, m0, P0, A, Q, H, R, 1);

    rts_m = [zeros(size(kf_m[1])) for i in 1:length(Y)];
    rts_P = [zeros(size(kf_P[1])) for i in 1:length(Y)];

    rts!(rts_m, rts_P, kf_m, kf_P, A, Q, 1);

    @time rts!(rts_m, rts_P, kf_m, kf_P, A, Q, niter);

    ms = rts_m[1]
    println("ms = ", ms)

    rmse_rts = 0.0;
    for k in 1:length(rts_m)
        rmse_rts = rmse_rts + sum((rts_m[k][1:2,1] - X[k][1:2,1]).^2)
    end
    rmse_rts = sqrt(rmse_rts / length(rts_m))

    println("rmse_rts = ", rmse_rts)
end

run_rts();
