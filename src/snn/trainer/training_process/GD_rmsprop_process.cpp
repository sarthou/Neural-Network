//============================================================================
// Name        : GD_rmsprop_process.cpp
// Authors     : Guillaume Sarthou
// EMail       : open.pode@gmail.com
// Date		   : 29 jun. 2017
// Version     : V1.5
// Copyright   : This file is part of SNN_network project which is released under
//               MIT license.
//============================================================================
#include "snn/trainer/training_process/GD_rmsprop_process.h"
#include <cmath>

namespace SNN
{

	GD_rmsprop_process::GD_rmsprop_process(Perceptron* p_perceptron, double p_step) : Trainig_process(p_perceptron)
	{
		m_step = p_step;
		m_Eg = 0;
	}

	GD_rmsprop_process::~GD_rmsprop_process()
	{

	}

	void GD_rmsprop_process::propagate(vector<Trainig_process*>* process, bool out)
	{
		if (out)
			m_gradient = -m_error*get_single_derivate();
		else
			m_gradient *= get_single_derivate();

		m_Eg = 0.9*m_Eg + 0.1*m_gradient*m_gradient;

		add_to_precedent(process, m_gradient);

		m_delta = m_step * m_gradient / sqrt(m_Eg + 0.0001);
	}

	void GD_rmsprop_process::set_default_configuration(trainig_config_t* configuration)
	{
		
		if (configuration->step == UNDEFINED)
			configuration->step = 0.002;
	}

} // namespace SNN_trainer