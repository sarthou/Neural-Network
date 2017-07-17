//============================================================================
// Name        : GD_rmsprop_process.h
// Authors     : Guillaume Sarthou
// EMail       : open.pode@gmail.com
// Date		   : 18 jul. 2017
// Version     : V2.0
// Copyright   : This file is part of SNN_network project which is released under
//               MIT license.
//============================================================================

#ifndef GD_RMSPROP_PROCESS_H
#define GD_RMSPROP_PROCESS_H

#include "snn/trainer/training_process/Trainig_process.h"
#include <cmath>

namespace SNN
{

	class GD_rmsprop_process : public Trainig_process
	{
	public:
		GD_rmsprop_process(Perceptron* p_perceptron, float p_step);
		~GD_rmsprop_process();

		void propagate(vector<Trainig_process*>* process, bool out = false);

		static void set_default_configuration(trainig_config_t* configuration);

	private:
		float m_step;

		float m_Eg;
	};
} // SNN_network

#endif // !GD_RMSPROP_PROCESS_H
