//============================================================================
// Name        : GD_adam_process.h
// Authors     : Guillaume Sarthou
// EMail       : open.pode@gmail.com
// Date		   : 11 jun. 2017
// Version     : V1.4
// Copyright   : This file is part of SNN_network project which is released under
//               MIT license.
//============================================================================
#pragma once

#ifndef GD_ADAM_PROCESS_H
#define GD_ADAM_PROCESS_H

#include "Trainig_process.h"

namespace SNN_network
{

	class GD_adam_process : public Trainig_process
	{
	public:
		GD_adam_process(Perceptron* p_perceptron, double p_step);
		~GD_adam_process();

		void propagate(vector<Trainig_process*> process, bool out = false);

		static void set_default_configuration(trainig_config_t* configuration);

	private:
		double m_step;

		double m_m;
		double m_v;

		float B1;
		float B2;
	};
} // SNN_network

#endif // !GD_ADAM_PROCESS_H
