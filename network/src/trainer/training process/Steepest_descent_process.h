#pragma once

#ifndef STEEPEST_DESCENT_PROCESS_H
#define STEEPEST_DESCENT_PROCESS_H

#include "Trainig_process.h"

class Steepest_descent_process : public Trainig_process
{
public:
	Steepest_descent_process(Perceptron* p_perceptron, double p_step);
	~Steepest_descent_process();

	void init();
	void set_error(double T);
	void propagate(vector<Trainig_process*> process, bool out = false);
	void compute();

	void add(double value);

private:
	double m_step;
	double m_bia_gradient;
	vector<double> m_w_gradient;

	double m_error;
	double m_delta;
};

#endif