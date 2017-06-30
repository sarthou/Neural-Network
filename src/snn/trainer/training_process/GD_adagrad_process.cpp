//============================================================================
// Name        : GD_adagrad_process.cpp
// Authors     : Guillaume Sarthou
// EMail       : open.pode@gmail.com
// Date		   : 29 jun. 2017
// Version     : V1.5
// Copyright   : This file is part of SNN_network project which is released under
//               MIT license.
//============================================================================
#include "snn/trainer/training_process/GD_adagrad_process.h"
#include <cmath>

namespace SNN
{

	GD_adagrad_process::GD_adagrad_process(Perceptron* p_perceptron, double p_step) : Trainig_process(p_perceptron)
	{
		m_step = p_step;
		ss_gradient = 0;
	}

	GD_adagrad_process::~GD_adagrad_process()
	{

	}

	void GD_adagrad_process::propagate(vector<Trainig_process*>* process, bool out)
	{
		derivate_perceptron();

		if (out)
			m_gradient = -m_error*get_derivate();
		else
			m_gradient *= get_derivate();

		ss_gradient += m_gradient*m_gradient;

		add_to_precedent(process, m_gradient);

		m_delta = m_gradient*m_step*20/ sqrt(ss_gradient + 0.00000001);
	}

	void GD_adagrad_process::set_default_configuration(trainig_config_t* configuration)
	{
		if(configuration->step == UNDEFINED)
			configuration->step = 0.05;
	}

} // namespace SNN_trainer