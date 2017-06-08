//============================================================================
// Name        : GD_adagrad_process.h
// Authors     : Guillaume Sarthou
// EMail       : open.pode@gmail.com
// Date		   : 8 jun. 2017
// Version     : V1.4
// Copyright   : This file is part of SNN_network project which is released under
//               MIT license.
//============================================================================
#pragma once

#ifndef GD_ADAGRAD_PROCESS_H
#define GD_ADAGRAD_PROCESS_H

#include "Trainig_process.h"

namespace SNN_network
{

	class GD_adagrad_process : public Trainig_process
	{
	public:
		GD_adagrad_process(Perceptron* p_perceptron, double p_step);
		~GD_adagrad_process();

		void propagate(vector<Trainig_process*> process, bool out = false);
		void compute();

	private:
		double m_step;
		double m_bia_gradient;
		vector<double> m_w_gradient;

		double ss_gradient;

		double m_delta;
	};
} // SNN_network

#endif // !GD_ADAGRAD_PROCESS_H
