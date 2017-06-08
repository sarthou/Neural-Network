//============================================================================
// Name        : Steepest_descent_process.cpp
// Authors     : Guillaume Sarthou
// EMail       : open.pode@gmail.com
// Date		   : 8 jun. 2017
// Version     : V1.4
// Copyright   : This file is part of SNN_network project which is released under
//               MIT license.
//============================================================================
#include "Steepest_descent_process.h"

namespace SNN_network
{

	Steepest_descent_process::Steepest_descent_process(Perceptron* p_perceptron, double p_step) : Trainig_process(p_perceptron)
	{
		m_step = p_step;
	}

	Steepest_descent_process::~Steepest_descent_process()
	{

	}

	void Steepest_descent_process::propagate(vector<Trainig_process*> process, bool out)
	{
		derivate_perceptron();

		if (out)
			m_gradient = -m_error*get_derivate();
		else
			m_gradient *= get_derivate();

		add_to_precedent(process, m_gradient);

		m_delta = m_gradient*m_step;
	}

} // namespace SNN_trainer