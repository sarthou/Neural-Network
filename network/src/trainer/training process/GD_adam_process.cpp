//============================================================================
// Name        : GD_adam_process.cpp
// Authors     : Guillaume Sarthou
// EMail       : open.pode@gmail.com
// Date		   : 25 jun. 2017
// Version     : V1.4
// Copyright   : This file is part of SNN_network project which is released under
//               MIT license.
//============================================================================
#include "GD_adam_process.h"

namespace SNN
{

	GD_adam_process::GD_adam_process(Perceptron* p_perceptron, double p_step) : Trainig_process(p_perceptron)
	{
		m_step = p_step;
		m_m = 0;
		m_v = 0;
		B1 = 0.9;
		B2 = 0.999;
	}

	GD_adam_process::~GD_adam_process()
	{

	}

	void GD_adam_process::propagate(vector<Trainig_process*> process, bool out)
	{
		derivate_perceptron();


		if (out)
			m_gradient = -m_error*get_derivate();
		else
			m_gradient *= get_derivate();

		m_m = ((1 - B1)*m_m /B1  + m_gradient) ;
		m_v = ((1 - B2)*m_v /B2  + m_gradient*m_gradient) ;

		add_to_precedent(process, m_gradient);

		m_delta = m_step * m_m / sqrt(m_v + 0.00000001);
	}

	void GD_adam_process::set_default_configuration(trainig_config_t* configuration)
	{
		if (configuration->step == INFINITY)
			configuration->step = 0.001;
	}

} // namespace SNN_trainer