//============================================================================
// Name        : GD_momentum_process.h
// Authors     : Guillaume Sarthou
// EMail       : open.pode@gmail.com
// Date		   : 25 jun. 2017
// Version     : V1.4
// Copyright   : This file is part of SNN_network project which is released under
//               MIT license.
//============================================================================
#pragma once

#ifndef GD_MOMENTUM_PROCESS_H
#define GD_MOMENTUM_PROCESS_H

#include "Trainig_process.h"

namespace SNN
{

	class GD_momentum_process : public Trainig_process
	{
	public:
		GD_momentum_process(Perceptron* p_perceptron, double p_step, double momentum_factor);
		~GD_momentum_process();

		void propagate(vector<Trainig_process*> process, bool out = false);

		static void set_default_configuration(trainig_config_t* configuration);

	private:
		double m_step;
		double m_momentum_factor;

		double m_delta_1;
	};
} // SNN_network

#endif // !GD_MOMENTUM_PROCESS_H
