//============================================================================
// Name        : Steepest_descent_process.h
// Authors     : Guillaume Sarthou
// EMail       : open.pode@gmail.com
// Date		   : 29 jun. 2017
// Version     : V1.5
// Copyright   : This file is part of SNN_network project which is released under
//               MIT license.
//============================================================================

#ifndef STEEPEST_DESCENT_PROCESS_H
#define STEEPEST_DESCENT_PROCESS_H

#include "snn/trainer/training_process/Trainig_process.h"
#include <cmath>

namespace SNN
{

	class Steepest_descent_process : public Trainig_process
	{
	public:
		Steepest_descent_process(Perceptron* p_perceptron, float p_step);
		~Steepest_descent_process();

		void propagate(vector<Trainig_process*>* process, bool out = false);

		static void set_default_configuration(trainig_config_t* configuration);

	private:
		float m_step;
	};

} // namespace SNN_network

#endif