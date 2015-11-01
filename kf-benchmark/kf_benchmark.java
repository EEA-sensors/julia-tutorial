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

import org.ejml.simple.SimpleMatrix;
import java.io.*;
import java.util.*;

public class kf_benchmark {
    public static void main(String[] args) throws IOException {

	//
	// Read data
	//
	List<SimpleMatrix> X = new ArrayList<SimpleMatrix>();
	List<SimpleMatrix> Y = new ArrayList<SimpleMatrix>();

	for (int i = 0; i < 2; i++) {
	    BufferedReader reader = null;
	    if (i == 0) {
		reader = new BufferedReader(new FileReader(new File("xdata.txt")));
	    }
	    else {
		reader = new BufferedReader(new FileReader(new File("ydata.txt")));
	    }

	    String line;
	    while ((line = reader.readLine()) != null) {
		String[] elem = line.trim().split(" +");
		if (elem.length > 0) {
		    SimpleMatrix tmp = new SimpleMatrix(elem.length, 1);
		    for (int j = 0; j < elem.length; j++) {
			tmp.set(j, 0, Double.parseDouble(elem[j]));
		    }
		    if (i == 0) {
			X.add(tmp);
		    }
		    else {
			Y.add(tmp);
		    }
		}
	    }
	}

	// System.out.println(Y);
	System.out.println("Read " + Y.size() + " observations.");

	//
	// Parameters
	//
	double q = 1.0;
	double dt = 0.1;
	double s = 0.5;
	SimpleMatrix A = new SimpleMatrix(4, 4, true,
					  1.0, 0, dt, 0,
					  0, 1.0, 0, dt,
					  0, 0, 1.0, 0,
					  0, 0, 0, 1.0);
	
	SimpleMatrix Q = new SimpleMatrix(4, 4, true,
					  q*dt*dt*dt/3, 0, q*dt*dt/2, 0,
					  0, q*dt*dt*dt/3, 0, q*dt*dt/2,
					  q*dt*dt/2, 0, q*dt, 0,
					  0, q*dt*dt/2, 0, q*dt);

	SimpleMatrix H = new SimpleMatrix(2, 4, true,
					  1.0, 0, 0, 0,
					  0, 1.0, 0, 0);
	SimpleMatrix R = SimpleMatrix.diag(s*s, s*s);
	SimpleMatrix m0 = new SimpleMatrix(4, 1, true,
					   0, 0, 1.0, -1.0);
	SimpleMatrix P0 = SimpleMatrix.diag(1,1,1,1);

	int niter = 10000;
	System.out.println("niter = " + niter);

	//
	// Kalman filter
	//
	List<SimpleMatrix> kf_m = new ArrayList<SimpleMatrix>(Y.size());
	List<SimpleMatrix> kf_P = new ArrayList<SimpleMatrix>(Y.size());

	SimpleMatrix m, P;
	m = m0.copy();
	P = P0.copy();

	long startTime = System.currentTimeMillis();

	for (int i = 0; i < niter; i++) {
	    kf_m.clear();
	    kf_P.clear();

	    m = m0.copy();
	    P = P0.copy();

	    for (SimpleMatrix y : Y) {
		m = A.mult(m);
		P = A.mult(P).mult(A.transpose()).plus(Q);

		SimpleMatrix S = H.mult(P).mult(H.transpose()).plus(R);
		SimpleMatrix K = S.solve(H.mult(P)).transpose();
		m = m.plus(K.mult(y.minus(H.mult(m))));
		P = P.minus(K.mult(S).mult(K.transpose()));

		kf_m.add(m);
		kf_P.add(P);
	    }
	}
	
	long endTime   = System.currentTimeMillis();
	double totalTime = (endTime - startTime) / 1000.0;
	System.out.println("Elapsed time " + totalTime + " seconds.");

	System.out.println("m = " + m);

	{
	    ListIterator<SimpleMatrix> m_iter = kf_m.listIterator(0);
	    ListIterator<SimpleMatrix> X_iter = X.listIterator(0);

	    double rmse_kf = 0.0;
	    
	    while(m_iter.hasNext()) {
		SimpleMatrix e = m_iter.next();
		SimpleMatrix x = X_iter.next();

		rmse_kf += (e.get(0,0) - x.get(0,0)) * (e.get(0,0) - x.get(0,0))
		    + (e.get(1,0) - x.get(1,0)) * (e.get(1,0) - x.get(1,0));
	    }

	    rmse_kf = Math.sqrt(rmse_kf / X.size());
	    System.out.println("rmse_kf = " + rmse_kf);
	}
	
	//
	// RTS smoother
	//
	List<SimpleMatrix> rts_m = new ArrayList<SimpleMatrix>(Y.size());
	List<SimpleMatrix> rts_P = new ArrayList<SimpleMatrix>(Y.size());

	SimpleMatrix ms, Ps;
	ms = m.copy();
	Ps = P.copy();
	
	startTime = System.currentTimeMillis();

	for (int i = 0; i < niter; i++) {
	    rts_m.clear();
	    rts_P.clear();

	    ms = m.copy();
	    Ps = P.copy();

	    rts_m.add(ms);
	    rts_P.add(Ps);
	    
	    ListIterator<SimpleMatrix> m_iter = kf_m.listIterator(kf_m.size() - 1);
	    ListIterator<SimpleMatrix> P_iter = kf_P.listIterator(kf_P.size() - 1);

	    while(m_iter.hasPrevious()) {
		SimpleMatrix mf = m_iter.previous();
		SimpleMatrix Pf = P_iter.previous();

		SimpleMatrix mp = A.mult(mf);
		SimpleMatrix Pp = A.mult(Pf).mult(A.transpose()).plus(Q);
		SimpleMatrix Ck = Pp.solve(A.mult(Pf)).transpose();
		ms = mf.plus(Ck.mult(ms.minus(mp)));
		Ps = Pf.plus(Ck.mult(Ps.minus(Pp)).mult(Ck.transpose()));
		rts_m.add(ms);
		rts_P.add(Ps);
	    }
	}

	endTime   = System.currentTimeMillis();
	totalTime = (endTime - startTime) / 1000.0;
	System.out.println("Elapsed time " + totalTime + " seconds.");

	Collections.reverse(rts_m);
	Collections.reverse(rts_P);
	
	System.out.println("ms = " + ms);

	{
	    ListIterator<SimpleMatrix> m_iter = rts_m.listIterator(0);
	    ListIterator<SimpleMatrix> X_iter = X.listIterator(0);

	    double rmse_rts = 0.0;
	    
	    while(m_iter.hasNext()) {
		SimpleMatrix e = m_iter.next();
		SimpleMatrix x = X_iter.next();

		rmse_rts += (e.get(0,0) - x.get(0,0)) * (e.get(0,0) - x.get(0,0))
		    + (e.get(1,0) - x.get(1,0)) * (e.get(1,0) - x.get(1,0));
	    }

	    rmse_rts = Math.sqrt(rmse_rts / X.size());
	    System.out.println("rmse_rts = " + rmse_rts);
	}
    }
}
