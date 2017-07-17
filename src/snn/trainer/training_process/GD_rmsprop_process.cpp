//============================================================================
// Name        : GD_rmsprop_process.cpp
// Authors     : Guillaume Sarthou
// EMail       : open.pode@gmail.com
// Date		   : 18 jul. 2017
// Version     : V2.0
// Copyright   : This file is part of SNN_network project which is released under
//               MIT license.
//============================================================================
#include "snn/trainer/training_process/GD_rmsprop_process.h"
#include <cmath>

namespace SNN
{

	GD_rmsprop_process::GD_rmsprop_process(Perceptron* p_perceptron, float p_step) : Trainig_process(p_perceptron)
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

		m_Eg = 0.9f*m_Eg + 0.1f*m_gradient*m_gradient;

		add_to_precedent(process, m_gradient);

		m_delta = m_step * m_gradient / sqrt(m_Eg + 0.0001f);
	}

	void GD_rmsprop_process::set_default_configuration(trainig_config_t* configuration)
	{
		
		if (configuration->step == UNDEFINED)
			configuration->step = 0.002f;
	}

} // namespace SNN_trainer