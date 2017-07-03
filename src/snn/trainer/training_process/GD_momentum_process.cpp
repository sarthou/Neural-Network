//============================================================================
// Name        : GD_momentum_process.cpp
// Authors     : Guillaume Sarthou
// EMail       : open.pode@gmail.com
// Date		   : 29 jun. 2017
// Version     : V1.5
// Copyright   : This file is part of SNN_network project which is released under
//               MIT license.
//============================================================================
#include "snn/trainer/training_process/GD_momentum_process.h"
#include <cmath>

namespace SNN
{

	GD_momentum_process::GD_momentum_process(Perceptron* p_perceptron, double p_step, double momentum_factor) : Trainig_process(p_perceptron)
	{
		m_step = p_step;
		m_delta_1 = 0;
		m_momentum_factor = momentum_factor;
	}

	GD_momentum_process::~GD_momentum_process()
	{

	}

	void GD_momentum_process::propagate(vector<Trainig_process*>* process, bool out)
	{
		m_delta_1 = m_gradient;

		if (out)
			m_gradient = -m_error*get_single_derivate();
		else
			m_gradient *= get_single_derivate();

		add_to_precedent(process, m_gradient);

		m_delta = m_gradient*m_step + m_momentum_factor*m_delta_1;
	}

	void GD_momentum_process::set_default_configuration(trainig_config_t* configuration)
	{
		if (configuration->step == UNDEFINED)
			configuration->step = 0.05;

		if (configuration->momentum_factor == UNDEFINED)
			configuration->momentum_factor = 0.05;
	}

} // namespace SNN_trainer