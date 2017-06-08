//============================================================================
// Name        : GD_adagrad_process.cpp
// Authors     : Guillaume Sarthou
// EMail       : open.pode@gmail.com
// Date		   : 8 jun. 2017
// Version     : V1.4
// Copyright   : This file is part of SNN_network project which is released under
//               MIT license.
//============================================================================
#include "GD_adagrad_process.h"

namespace SNN_network
{

	GD_adagrad_process::GD_adagrad_process(Perceptron* p_perceptron, double p_step) : Trainig_process(p_perceptron)
	{
		m_step = p_step;
		m_delta = 0;
		m_error = 0;
		m_gradient = 0;
		ss_gradient = 0;
		m_w_gradient.resize(m_perceptron->get_weigh().size());
	}

	GD_adagrad_process::~GD_adagrad_process()
	{

	}

	void GD_adagrad_process::propagate(vector<Trainig_process*> process, bool out)
	{
		derivate_perceptron();

		if (out)
			m_gradient = -m_error*get_derivate();
		else
			m_gradient *= get_derivate();

		ss_gradient += m_gradient*m_gradient;

		add_to_precedent(process, m_gradient);

		m_delta = m_gradient*m_step*20/ sqrtf(ss_gradient + 0.00000001);
	}

} // namespace SNN_trainer