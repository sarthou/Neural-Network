//============================================================================
// Name        : GD_rmsprop_process.cpp
// Authors     : Guillaume Sarthou
// EMail       : open.pode@gmail.com
// Date		   : 7 jun. 2017
// Version     : V1.3
// Copyright   : This file is part of SNN_network project which is released under
//               MIT license.
//============================================================================
#include "GD_rmsprop_process.h"

namespace SNN_network
{

	GD_rmsprop_process::GD_rmsprop_process(Perceptron* p_perceptron, double p_step) : Trainig_process(p_perceptron)
	{
		m_step = p_step;
		m_Eg = 0;
	}

	GD_rmsprop_process::~GD_rmsprop_process()
	{

	}

	void GD_rmsprop_process::propagate(vector<Trainig_process*> process, bool out)
	{
		derivate_perceptron();


		if (out)
			m_gradient = -m_error*get_derivate();
		else
			m_gradient *= get_derivate();

		m_Eg = 0.9*m_Eg + 0.1*m_gradient*m_gradient;

		add_to_precedent(process, m_gradient);

		m_delta = m_step * m_gradient / sqrtf(m_Eg + 0.0001);
	}

} // namespace SNN_trainer