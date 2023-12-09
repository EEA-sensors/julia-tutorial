/*
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
*/
				  
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <ctime>
#include <Eigen/Dense>

int main(void) {
  size_t niter = 10000;

  std::cout << "niter = " << niter << std::endl;
      
  //
  // Read the data
  //
  std::vector< Eigen::Vector4d > X;
  std::vector< Eigen::Vector2d > Y;
  
  std::ifstream xfile("xdata.txt");
  if (!xfile.is_open()) {
    std::cerr << "Oops, cannot read X." << std::endl;
    return 1;
  }
  
  std::ifstream yfile("ydata.txt");
  if (!yfile.is_open()) {
    std::cerr << "Oops, cannot read Y." << std::endl;
    return 1;
  }
  
  while (!xfile.eof()) {
    Eigen::Vector4d x;
    xfile >> x[0];
    xfile >> x[1];
    xfile >> x[2];
    xfile >> x[3];

    if (!xfile.eof()) {
      X.push_back(x);
      // std::cout << x << std::endl << std::endl;
    }
  }
  xfile.close();
  
  while (!yfile.eof()) {
    Eigen::Vector2d y;
    yfile >> y[0];
    yfile >> y[1];

    if (!yfile.eof()) {
      Y.push_back(y);
      // std::cout << y << std::endl << std::endl;
    }
  }
  yfile.close();

  std::cout << "Size of X is " << X.size() << std::endl;
  std::cout << "Size of Y is " << Y.size() << std::endl;
    
  //
  // Parameters
  //
  double q = 1;
  double dt = 0.1;
  double s = 0.5;
  
  Eigen::Matrix4d A;
  A <<
    1, 0, dt, 0,
    0, 1, 0, dt,
    0, 0, 1, 0,
    0, 0, 0, 1;
  
  Eigen::Matrix4d Q;
  Q <<
    q*dt*dt*dt/3, 0, q*dt*dt/2, 0,
    0, q*dt*dt*dt/3, 0, q*dt*dt/2,
    q*dt*dt/2, 0, q*dt, 0,
    0, q*dt*dt/2, 0, q*dt;
    
  Eigen::Matrix<double,2,4> H;
  H <<
    1, 0, 0, 0,
    0, 1, 0, 0;
  
  Eigen::Matrix2d R = s * s * Eigen::Matrix2d::Identity();

  Eigen::Vector4d m0;
  m0 << 0, 0, 1, -1;

  Eigen::Matrix4d P0 = Eigen::Matrix4d::Identity();

  //
  // Kalman filter
  //
  Eigen::Vector4d kf_m[Y.size()];
  Eigen::Matrix4d kf_P[Y.size()];

  Eigen::Vector4d m;
  Eigen::Matrix4d P;

  Eigen::Matrix4d At = A.transpose();
  Eigen::Matrix<double,4,2> Ht = H.transpose();

  clock_t begin = clock();
  
  for (size_t i = 0; i < niter; i++) {
    m = m0;
    P = P0;
    
    for (int k = 0; k < Y.size(); k++) {
      m = A*m;
      P = A*P*At + Q;

      Eigen::Matrix2d S = H*P*Ht + R;
      Eigen::LDLT<Eigen::Matrix2d> Sdecomp = S.ldlt();
      Eigen::Matrix<double,4,2> K = Sdecomp.solve(H * P).transpose();
      
      m = m + K*(Y[k] - H*m);
      P = P - K*S*K.transpose();

      kf_m[k] = m;
      kf_P[k] = P;
    }
  }

  clock_t end = clock();
  double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

  std::cout << "Elapsed time is " << elapsed_secs << " seconds." << std::endl;
  
  std::cout << "m = " << std::endl << m << std::endl;

  double rmse_kf = 0;
  
  for (int k = 0; k < X.size(); k++) {
    rmse_kf += (X[k][0] - kf_m[k][0]) * (X[k][0] - kf_m[k][0])
      + (X[k][1] - kf_m[k][1]) * (X[k][1] - kf_m[k][1]);
  }
  rmse_kf = sqrt(rmse_kf / X.size());

  std::cout << "rmse_kf = " << rmse_kf << std::endl;
  
  //
  // RTS smoother
  //

  Eigen::Vector4d rts_m[Y.size()];
  Eigen::Matrix4d rts_P[Y.size()];

  Eigen::Vector4d ms;
  Eigen::Matrix4d Ps;

  begin = clock();
  
  for (size_t i = 0; i < niter; i++) {
    ms = m;
    Ps = P;

    rts_m[Y.size()-1] = ms;
    rts_P[Y.size()-1] = Ps;

    for (int k = Y.size()-2; k >= 0; k--) {
      Eigen::Vector4d mp = A*kf_m[k];
      Eigen::Matrix4d Pp = A*kf_P[k]*At + Q;
      Eigen::Matrix4d Ck = Pp.ldlt().solve(A * kf_P[k]).transpose();
      ms = kf_m[k] + Ck*(ms - mp);
      Ps = kf_P[k] + Ck*(Ps - Pp)*Ck.transpose();
      rts_m[Y.size()-1-k] = ms;
      rts_P[Y.size()-1-k] = Ps;
    }
  }

  end = clock();
  elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

  std::cout << "Elapsed time is " << elapsed_secs << " seconds." << std::endl;

  double rmse_rts = 0;
  
  for (int k = 0; k < X.size(); k++) {
    rmse_rts += (X[k][0] - rts_m[k][0]) * (X[k][0] - rts_m[k][0])
      + (X[k][1] - rts_m[k][1]) * (X[k][1] - rts_m[k][1]);
  }
  rmse_rts = sqrt(rmse_rts / X.size());

  std::cout << "rmse_rts = " << rmse_rts << std::endl;

  std::cout << "ms = " << std::endl << ms << std::endl;

  return 0;    
}
