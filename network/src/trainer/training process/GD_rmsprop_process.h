//============================================================================
// Name        : GD_rmsprop_process.h
// Authors     : Guillaume Sarthou
// EMail       : open.pode@gmail.com
// Date		   : 11 jun. 2017
// Version     : V1.4
// Copyright   : This file is part of SNN_network project which is released under
//               MIT license.
//============================================================================
#pragma once

#ifndef GD_RMSPROP_PROCESS_H
#define GD_RMSPROP_PROCESS_H

#include "Trainig_process.h"

namespace SNN_network
{

	class GD_rmsprop_process : public Trainig_process
	{
	public:
		GD_rmsprop_process(Perceptron* p_perceptron, double p_step);
		~GD_rmsprop_process();

		void propagate(vector<Trainig_process*> process, bool out = false);

		static void set_default_configuration(trainig_config_t* configuration);

	private:
		double m_step;

		double m_Eg;
	};
} // SNN_network

#endif // !GD_RMSPROP_PROCESS_H
