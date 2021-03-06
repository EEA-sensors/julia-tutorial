
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
    R = s^2*eye(2);
    m0 = [0;0;1.0;-1.0];
    P0 = eye(4);

    niter = 10000
    println("niter = ", niter)

#
# Kalman filter
#

    kf_m = [zeros(size(m0)) for i in 1:length(Y)];
    kf_P = [zeros(size(P0)) for i in 1:length(Y)];

    m = m0;
    P = P0;

    tic()
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
    toc()

    println("m = ", m)

    rmse_kf = 0.0;
    for k in 1:length(kf_m)
        rmse_kf = rmse_kf + sum((kf_m[k][1:2,1] - X[k][1:2,1]).^2)
    end
    rmse_kf = sqrt(rmse_kf / length(kf_m))

    println("rmse_kf = ", rmse_kf)

#
# RTS smoother
#

    rts_m = [zeros(size(m0)) for i in 1:length(Y)];
    rts_P = [zeros(size(P0)) for i in 1:length(Y)];
    
    ms = m;
    Ps = P;

    tic()
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
    toc()
    
    println("ms = ", ms)

    rmse_rts = 0.0;
    for k in 1:length(rts_m)
        rmse_rts = rmse_rts + sum((rts_m[k][1:2,1] - X[k][1:2,1]).^2)
    end
    rmse_rts = sqrt(rmse_rts / length(rts_m))
    
    println("rmse_rts = ", rmse_rts)

