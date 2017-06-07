//============================================================================
// Name        : Steepest_descent_process.h
// Authors     : Guillaume Sarthou
// EMail       : open.pode@gmail.com
// Date		   : 7 jun. 2017
// Version     : V1.3
// Copyright   : This file is part of SNN_network project which is released under
//               MIT license.
//============================================================================
#pragma once

#ifndef STEEPEST_DESCENT_PROCESS_H
#define STEEPEST_DESCENT_PROCESS_H

#include "Trainig_process.h"

namespace SNN_network
{

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
		double m_gradient;
	};

} // namespace SNN_network

#endif