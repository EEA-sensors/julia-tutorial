
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


#
# Run with `julia --project=. -O2 kf_benchmark_improved.jl`
#

using DelimitedFiles
using StaticArrays
using LinearAlgebra

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
eye(n) = diagm(ones(n))

q = 1.0;
dt = 0.1;
s = 0.5;
A = SArray{Tuple{4,4}}([1.0 0 dt 0;
    0 1.0 0 dt;
    0 0 1.0 0;
    0 0 0 1.0]);
Q = SArray{Tuple{4,4}}(q*[dt^3/3 0 dt^2/2 0;
    0 dt^3/3 0 dt^2/2;
    dt^2/2 0 dt 0;
    0 dt^2/2 0 dt]);

H = SArray{Tuple{2,4}}([1.0 0 0 0;
    0 1.0 0 0]);

R = SArray{Tuple{2,2}}(s^2*eye(2));

m0 = SArray{Tuple{4}}([0;0;1.0;-1.0]);

P0 = SArray{Tuple{4,4}}(eye(4));

niter = 10000
println("niter = ", niter)

#
# Kalman filter
#

kf_m = [SArray{Tuple{4}}(zeros(size(m0))) for i in 1:length(Y)];
kf_P = [SArray{Tuple{4,4}}(zeros(size(P0))) for i in 1:length(Y)];

function kfilt!(x, y, niter, m0, P0, A, Q, H, Y, R)
    m = SArray{Tuple{4}}(m0)
    P = SArray{Tuple{4,4}}(P0)
    At = A'
    Ht = H'
    for _ in 1:niter
        m = m0
        P = P0
        for k in 1:length(Y)

            m = A*m;
            P = A*P*At + Q;

            S = H*P*Ht + R;
            K = P*Ht/S;
            D = Y[k]-H*m
            m = m+K*D;
            P = P - K*S*K';

            @inbounds x[k] = m;
            @inbounds y[k] = P;
        end
    end
end


x = kf_m
y = kf_P
# warmup
kfilt!(x, y, 1, m0, P0, A, Q, H,Y, R)

x = kf_m
y = kf_P

@time kfilt!(x, y, niter, m0, P0, A, Q, H,Y, R)

rmse_kf =
    mapreduce(+, 1:length(x)) do k
        sum((x[k][1:2,1] - X[k][1:2,1]).^2)
    end
rmse_kf = sqrt(rmse_kf / length(kf_m))

println("rmse_kf = ", rmse_kf)
@info x[end]


#
# RTS smoother
#
function rts!(x, y, kf_m, kf_P, A, Q, niter)
    m = kf_m[end];
    P = kf_P[end];
    At = A';

    for i in 1:niter
        ms = m;
        Ps = P;    
        rts_m[end] = ms;
        rts_P[end] = Ps;
        for k in length(kf_m)-1:-1:1
            mp = A*kf_m[k];
            Pp = A*kf_P[k]*At+Q;
            Ck = kf_P[k]*At/Pp; 
            ms = kf_m[k] + Ck*(ms - mp);
            Ps = kf_P[k] + Ck*(Ps - Pp)*Ck';
            @inbounds x[k] = ms;
            @inbounds y[k] = Ps;
        end
    end
end

rts_m = [SArray{Tuple{4}}(zeros(size(m0))) for i in 1:length(Y)];
rts_P = [SArray{Tuple{4,4}}(zeros(size(P0))) for i in 1:length(Y)];

x = rts_m
y = rts_P

rts!(x, y, kf_m, kf_P, A, Q, 1); # warmup

x = rts_m
y = rts_P

@time rts!(x, y, kf_m, kf_P, A, Q, niter)

rmse_rts =
    mapreduce(+, 1:length(x)) do k
        sum((x[k][1:2,1] - X[k][1:2,1]).^2)
    end
rmse_rts = sqrt(rmse_rts / length(rts_m))

println("rmse_rts = ", rmse_rts)
@info x[1]
