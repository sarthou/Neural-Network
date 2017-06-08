//============================================================================
// Name        : Steepest_descent_process.h
// Authors     : Guillaume Sarthou
// EMail       : open.pode@gmail.com
// Date		   : 8 jun. 2017
// Version     : V1.4
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

		void propagate(vector<Trainig_process*> process, bool out = false);

	private:
		double m_step;
	};

} // namespace SNN_network

#endif