//============================================================================
// Name        : Steepest_descent_process.cpp
// Authors     : Guillaume Sarthou
// EMail       : open.pode@gmail.com
// Date		   : 29 jun. 2017
// Version     : V1.5
// Copyright   : This file is part of SNN_network project which is released under
//               MIT license.
//============================================================================
#include "snn/trainer/training_process/Steepest_descent_process.h"
#include <cmath>

namespace SNN
{

	Steepest_descent_process::Steepest_descent_process(Perceptron* p_perceptron, float p_step) : Trainig_process(p_perceptron)
	{
		m_step = p_step;
	}

	Steepest_descent_process::~Steepest_descent_process()
	{

	}

	void Steepest_descent_process::propagate(vector<Trainig_process*>* process, bool out)
	{
		if (out)
			m_gradient = -m_error*get_single_derivate();
		else
			m_gradient *= get_single_derivate();

		add_to_precedent(process, m_gradient);

		m_delta = m_gradient*m_step;
	}

	void Steepest_descent_process::set_default_configuration(trainig_config_t* configuration)
	{
		if (configuration->step == UNDEFINED)
			configuration->step = 0.1f;
	}

} // namespace SNN_trainer