#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#define _USE_MATH_DEFINES
#include <math.h>
#include <cmath>
#include <chrono> 

using namespace std;

//Normal CDF
double normalCDF(double x){
	return erfc(-x / sqrt(2)) / 2;
}

//Normal PDF
double normalPDF(double x) {
	return (1 / sqrt(2 * M_PI) * exp(-pow(x,2)*0.5));
}

random_device rd;  //Will be used to obtain a seed for the random number engine
mt19937 generator(rd());	//default_random_engine generator;
normal_distribution<double> distribution(0.0, 1.0); // Standard Normal Dist.



class Call_EU {
	double S0, K, r, T, sigma;
	double d1 = (1 / (sigma * sqrt(T)) * (log(S0 / K) + (r + 0.5 * sigma * sigma) * T));
	double d2 = d1 - sigma * sqrt(T);


	vector<double> S_T;
	vector<double> payoff_vec;
	double payoff_mc = 0.0;
	double delta_mc_pw = 0.0;
	double delta_mc_lr = 0.0;
	double vega_mc_pw = 0.0;
	double vega_mc_lr = 0.0;
	double rho_mc_pw = 0.0;
	double rho_mc_lr = 0;
	double gamma_mc_lr_lr = 0.0;
	double gamma_mc_lr_pw = 0.0;
	double gamma_mc_pw_lr = 0.0;
	double sum_square_price = 0.0;
	double sum_square_delta_pw = 0.0;
	double sum_square_vega_pw = 0.0;
	double sum_square_rho_pw = 0.0;
	double sum_square_delta_lr = 0.0;
	double sum_square_vega_lr = 0.0;
	double sum_square_rho_lr = 0.0;
	double sum_square_gamma_lr_lr = 0.0;
	double sum_square_gamma_lr_pw = 0.0;
	double sum_square_gamma_pw_lr = 0.0;
	unsigned long long int m=0;
	double time_taken = 0;


public:
	Call_EU(double s, double k, double R, double t, double sig) : S0(s), K(k), r(R), T(t), sigma(sig) {};

	void MonteCarlo(unsigned long long int M) {
		S_T.clear();
		payoff_vec.clear();
		m = M;
		auto start = chrono::high_resolution_clock::now();

		for (int i = 0; i < M; i++) {
			double Z = distribution(generator);
			double S_T_i = S0 * exp((r - 0.5 * sigma * sigma) * T + sigma * sqrt(T) * Z);

			S_T.push_back(S_T_i);

			if (S_T_i > K) {
				payoff_vec.push_back(S_T_i - K);
				payoff_mc += exp(-r * T) * (S_T_i - K);
				
				delta_mc_pw += exp(-r * T) * S_T_i / S0;
				delta_mc_lr += exp(-r * T) * ((S_T_i - K) * Z) / (S0 * sigma * sqrt(T));
				
				vega_mc_pw += exp(-r * T) * (Z * sqrt(T) - sigma * T) * S_T_i;
				vega_mc_lr += exp(-r * T) * (S_T_i - K) * ((Z * Z / sigma) - (Z * sqrt(T)) - (1/sigma));
				
				gamma_mc_lr_lr += exp(-r * T) * (S_T_i - K) * ((pow(Z, 2) - 1) / (pow(S0, 2) * pow(sigma, 2) * T) - Z / (pow(S0, 2) * sigma * sqrt(T)));
				gamma_mc_lr_pw += exp(-r * T) * K * Z / (S0 * S0 * sigma * sqrt(T));
				gamma_mc_pw_lr += exp(-r * T) * (S_T_i / (S0*S0)) * ((Z / (sigma * sqrt(T)) - 1));
				
				rho_mc_pw += exp(-r * T) * K * T;
				rho_mc_lr += exp(-r * T) * (S_T_i - K) * ((Z * sqrt(T) / sigma) - T);
				
				sum_square_price += pow((S_T_i - K)*exp(-r*T),2);
				sum_square_delta_pw += pow((S_T_i / S0) * exp(-r*T), 2);
				sum_square_vega_pw += pow(((Z * sqrt(T) - sigma * T) * S_T_i) * exp(-r*T), 2);
				sum_square_rho_pw += pow(exp(-r * T) * S_T_i * T, 2);

				sum_square_gamma_lr_pw += pow(exp(-r * T) * K * Z / (S0 * S0 * sigma * sqrt(T)), 2);
				sum_square_gamma_lr_lr += pow(exp(-r * T) * (S_T_i - K) * ((pow(Z, 2) - 1) / (pow(S0, 2) * pow(sigma, 2) * T) - Z / (pow(S0, 2) * sigma * sqrt(T))), 2);
				sum_square_gamma_pw_lr += pow(exp(-r * T) * (S_T_i / (S0*S0)) * ((Z / (sigma * sqrt(T)) - 1)), 2);
				
				sum_square_delta_lr += pow(exp(-r * T) * ((S_T_i - K) * Z) / (S0 * sigma * sqrt(T)), 2);
				sum_square_vega_lr += pow(exp(-r * T) * (S_T_i - K) * ((Z * Z / sigma) - (Z * sqrt(T)) - (1 / sigma)), 2);
				sum_square_rho_lr += pow(exp(-r * T) * (S_T_i - K) * ((Z * sqrt(T) / sigma) - T),2);
			}
			else {
				payoff_vec.push_back(0);
			}
		}

		payoff_mc /= M;
		delta_mc_pw /=   M;
		delta_mc_lr /=  M;
		vega_mc_pw /= M;
		vega_mc_lr /=  M;
		rho_mc_pw /= M;
		rho_mc_lr /= M;
		gamma_mc_lr_lr /= M;
		gamma_mc_lr_pw /= M;
		gamma_mc_pw_lr /= M;
		sum_square_price /= (M*(M-1));
		sum_square_delta_pw /= (M * (M - 1));
		sum_square_vega_pw /= (M * (M - 1));
		sum_square_rho_pw /= (M * (M - 1));
		sum_square_delta_lr /= (M * (M - 1));
		sum_square_vega_lr /= (M * (M - 1));
		sum_square_rho_lr /= (M * (M - 1));
		sum_square_gamma_lr_lr /= (M * (M - 1));
		sum_square_gamma_lr_pw /= (M * (M - 1));
		sum_square_gamma_pw_lr/= (M * (M - 1));

		auto end = chrono::high_resolution_clock::now();
		time_taken = chrono::duration_cast<chrono::nanoseconds>(end - start).count();
	}

	double price_BS() {
		return S0 * normalCDF(d1) - K * exp(-r * T) * normalCDF(d2);
	}

	double delta_bs(){
		return normalCDF(d1);
	}

	double gamma_bs() {
		return (1 / (sigma * S0 * sqrt(T))) * normalPDF(d1);
	}

	double vega_bs() {
		return S0 * sqrt(T) * normalPDF(d1);
	}

	double rho_bs() {
		return K * T * exp(-r*T) * normalCDF(d2);
	}

	double price_mc() {
		return payoff_mc;
	}

	double bias() {
		return price_BS()-price_mc();
	}

	double variance_price() {
		return sum_square_price - (pow(payoff_mc,2) / (m-1)) ;
	}

	double variance_delta_pw() {
		return sum_square_delta_pw - (pow(delta_mc_pw, 2) / (m - 1));
	}

	double variance_vega_pw() {
		return sum_square_vega_pw - (pow(vega_mc_pw, 2) / (m - 1));
	}

	double variance_rho_pw() {
		return sum_square_rho_pw - (pow(rho_mc_pw, 2) / (m - 1));
	}
	
	double variance_delta_lr() {
		return sum_square_delta_lr - (pow(delta_mc_lr, 2) / (m - 1));
	}

	double variance_vega_lr() {
		return sum_square_vega_lr - (pow(vega_mc_lr, 2) / (m - 1));
	}

	double variance_rho_lr() {
		return sum_square_rho_lr - (pow(rho_mc_lr, 2) / (m - 1));
	}
	
	double variance_gamma_lr_lr() {
		return sum_square_gamma_lr_lr - (pow(gamma_mc_lr_lr, 2) / (m - 1));
	}

	double variance_gamma_lr_pw() {
		return sum_square_gamma_lr_pw - (pow(gamma_mc_lr_pw, 2) / (m - 1));
	}

	double variance_gamma_pw_lr() {
		return sum_square_gamma_pw_lr - (pow(gamma_mc_pw_lr, 2) / (m - 1));
	}

	double get_delta_mc_pw() {
		return delta_mc_pw;
	}

	double get_delta_mc_lr() {
		return delta_mc_lr;
	}

	double get_vega_mc_pw() {
		return vega_mc_pw;
	}

	double get_vega_mc_lr() {
		return vega_mc_lr;
	}

	double get_rho_mc_pw() {
		return rho_mc_pw;
	}

	double get_rho_mc_lr() {
		return  rho_mc_lr ;
	}

	double gamma_lr_lr() {
		return gamma_mc_lr_lr;
	}

	double gamma_lr_pw() {
		return gamma_mc_lr_pw;
	}

	double gamma_pw_lr() {
		return gamma_mc_pw_lr;
	}

	double time() {
		return time_taken;
	}

	void Statistics() {
		cout << "Number of Simulations: " << m << endl;
		cout << "Computation Time: " << time_taken << endl;
		cout << "" << endl;

		cout << "Statistics of Call Option: Monte Carlo - Price:" << endl;
		cout << "Estimate of Call Option Price: " << price_mc() << endl;
		cout << "Actual Call Option Price " << price_BS() << endl;
		cout << "Bias of estimator: " << bias() << endl;
		cout << "Variance of estimator: " << variance_price() << endl;
		cout << "Mean Square Error: " << variance_price() + pow(bias(), 2)<<endl;
		cout << "99.7% Confidence Interval [" << price_mc() - 3 * sqrt(variance_price())<<", "<< price_mc() + 3 * sqrt(variance_price())<<"]"<<endl;
		cout << "" << endl;
		
		//Pathwise Statistics
		cout << "Statistics of Call Option: Monte Carlo - Delta(PW):" << endl;
		cout << "Estimate of Call Option Delta: " << delta_mc_pw << endl;
		cout << "Actual Call Option Delta " << delta_bs() << endl;
		cout << "Bias of estimator: " << delta_bs()-delta_mc_pw << endl;
		cout << "Variance of estimator: " << variance_delta_pw() << endl;
		cout << "Mean Square Error: " << variance_price() + pow((delta_bs()-delta_mc_pw), 2) << endl;
		cout << "99.7% Confidence Interval [" << delta_mc_pw - 3 * sqrt(variance_delta_pw()) << ", " << delta_mc_pw + 3 * sqrt(variance_delta_pw()) << "]" << endl;
		cout << "" << endl;
		
		cout << "Statistics of Call Option: Monte Carlo - Delta(LR):" << endl;
		cout << "Estimate of Call Option Delta: " << delta_mc_lr << endl;
		cout << "Actual Call Option Delta " << delta_bs() << endl;
		cout << "Bias of estimator: " << delta_bs() - delta_mc_lr << endl;
		cout << "Variance of estimator: " << variance_delta_lr() << endl;
		cout << "Mean Square Error: " << variance_price() + pow((delta_bs() - delta_mc_lr), 2) << endl;
		cout << "99.7% Confidence Interval [" << delta_mc_lr - 3 * sqrt(variance_delta_lr()) << ", " << delta_mc_lr + 3 * sqrt(variance_delta_lr()) << "]" << endl;
		cout << "" << endl;
		
		cout << "Statistics of Call Option: Monte Carlo - Vega(PW):" << endl;
		cout << "Estimate of Call Option Vega: " << vega_mc_pw << endl;
		cout << "Actual Call Option Vega " << vega_bs() << endl;
		cout << "Bias of estimator: " << vega_bs() - vega_mc_pw << endl;
		cout << "Variance of estimator: " << variance_vega_pw() << endl;
		cout << "Mean Square Error: " << variance_vega_pw() + pow((vega_bs() - vega_mc_pw), 2) << endl;
		cout << "99.7% Confidence Interval [" << vega_mc_pw - 3 * sqrt(variance_vega_pw()) << ", " << vega_mc_pw + 3 * sqrt(variance_vega_pw()) << "]" << endl;
		cout << "" << endl;

		cout << "Statistics of Call Option: Monte Carlo - Vega(LR):" << endl;
		cout << "Estimate of Call Option Vega: " << vega_mc_lr << endl;
		cout << "Actual Call Option Vega " << vega_bs() << endl;
		cout << "Bias of estimator: " << vega_bs() - vega_mc_lr << endl;
		cout << "Variance of estimator: " << variance_vega_lr() << endl;
		cout << "Mean Square Error: " << variance_vega_lr() + pow((vega_bs() - vega_mc_lr), 2) << endl;
		cout << "99.7% Confidence Interval [" << vega_mc_lr - 3 * sqrt(variance_vega_lr()) << ", " << vega_mc_lr + 3 * sqrt(variance_vega_lr()) << "]" << endl;
		cout << "" << endl;

		cout << "Statistics of Call Option: Monte Carlo - Rho(PW):" << endl;
		cout << "Estimate of Call Option Rho: " << get_rho_mc_pw() << endl;
		cout << "Actual Call Option Rho " << rho_bs() << endl;
		cout << "Bias of estimator: " << rho_bs() - get_rho_mc_pw() << endl;
		cout << "Variance of estimator: " << variance_rho_pw() << endl;
		cout << "Mean Square Error: " << variance_rho_pw() + pow((rho_bs() - get_rho_mc_pw()), 2) << endl;
		cout << "99.7% Confidence Interval [" << get_rho_mc_pw() - 3 * sqrt(variance_rho_pw()) << ", " << get_rho_mc_pw() + 3 * sqrt(variance_rho_pw()) << "]" << endl;
		cout << "" << endl;
		
		cout << "Statistics of Call Option: Monte Carlo - Rho(LR):" << endl;
		cout << "Estimate of Call Option Rho: " << get_rho_mc_lr() << endl;
		cout << "Actual Call Option Rho " << rho_bs() << endl;
		cout << "Bias of estimator: " << rho_bs() - get_rho_mc_lr() << endl;
		cout << "Variance of estimator: " << variance_rho_lr() << endl;
		cout << "Mean Square Error: " << variance_rho_lr() + pow((rho_bs() - get_rho_mc_lr()), 2) << endl;
		cout << "99.7% Confidence Interval [" << get_rho_mc_lr() - 3 * sqrt(variance_rho_lr()) << ", " << get_rho_mc_lr() + 3 * sqrt(variance_rho_lr()) << "]" << endl;
		cout << "" << endl;
		//Gamma
		cout << "Statistics of Call Option: Monte Carlo - Gamma(LR-LR):" << endl;
		cout << "Estimate of Call Option Gamma: " << gamma_mc_lr_lr << endl;
		cout << "Actual Call Option Gamma " << gamma_bs() << endl;
		cout << "Bias of estimator: " << gamma_bs() - gamma_mc_lr_lr << endl;
		cout << "Variance of estimator: " << variance_gamma_lr_lr() << endl;
		cout << "Mean Square Error: " << variance_gamma_lr_lr() + pow((gamma_bs() - gamma_mc_lr_lr), 2) << endl;
		cout << "99.7% Confidence Interval [" << gamma_mc_lr_lr - 3 * sqrt(variance_gamma_lr_lr()) << ", " << gamma_mc_lr_lr + 3 * sqrt(variance_gamma_lr_lr()) << "]" << endl;
		cout << "" << endl;

		cout << "Statistics of Call Option: Monte Carlo - Gamma(LR-PW):" << endl;
		cout << "Estimate of Call Option Gamma: " << gamma_mc_lr_pw << endl;
		cout << "Actual Call Option Gamma " << gamma_bs() << endl;
		cout << "Bias of estimator: " << gamma_bs() - gamma_mc_lr_pw << endl;
		cout << "Variance of estimator: " << variance_gamma_lr_pw() << endl;
		cout << "Mean Square Error: " << variance_gamma_lr_pw() + pow((gamma_bs() - gamma_mc_lr_pw), 2) << endl;
		cout << "99.7% Confidence Interval [" << gamma_mc_lr_pw - 3 * sqrt(variance_gamma_lr_pw()) << ", " << gamma_mc_lr_pw + 3 * sqrt(variance_gamma_lr_pw()) << "]" << endl;
		cout << "" << endl;

		cout << "Statistics of Call Option: Monte Carlo - Gamma(PW-LR):" << endl;
		cout << "Estimate of Call Option Gamma: " << gamma_mc_pw_lr << endl;
		cout << "Actual Call Option Gamma " << gamma_bs() << endl;
		cout << "Bias of estimator: " << gamma_bs() - gamma_mc_pw_lr << endl;
		cout << "Variance of estimator: " << variance_gamma_pw_lr() << endl;
		cout << "Mean Square Error: " << variance_gamma_pw_lr() + pow((gamma_bs() - gamma_mc_pw_lr), 2) << endl;
		cout << "99.7% Confidence Interval [" << gamma_mc_pw_lr - 3 * sqrt(variance_gamma_pw_lr()) << ", " << gamma_mc_pw_lr + 3 * sqrt(variance_gamma_pw_lr()) << "]" << endl;
		cout << "" << endl;
	}
};


int main() {

	ofstream ofile("price.txt"); // creates an ofstream called ofile
	if (!ofile) {
		cout << "error opening file";
		exit(1); // error opening file
	}

	Call_EU s(100, 100, 0.05,1, 0.4);

	s.MonteCarlo(1000);
	s.Statistics();
		return 0;
}


