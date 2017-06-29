//============================================================================
// Name        : GD_adam_process.h
// Authors     : Guillaume Sarthou
// EMail       : open.pode@gmail.com
// Date		   : 29 jun. 2017
// Version     : V1.5
// Copyright   : This file is part of SNN_network project which is released under
//               MIT license.
//============================================================================

#ifndef GD_ADAM_PROCESS_H
#define GD_ADAM_PROCESS_H

#include "snn/trainer/training_process/Trainig_process.h"
#include <cmath>

namespace SNN
{

	class GD_adam_process : public Trainig_process
	{
	public:
		GD_adam_process(Perceptron* p_perceptron, double p_step);
		~GD_adam_process();

		void propagate(vector<Trainig_process*>* process, bool out = false);

		static void set_default_configuration(trainig_config_t* configuration);

	private:
		double m_step;

		double m_m;
		double m_v;

		double B1;
		double B2;
	};
} // SNN_network

#endif // !GD_ADAM_PROCESS_H
