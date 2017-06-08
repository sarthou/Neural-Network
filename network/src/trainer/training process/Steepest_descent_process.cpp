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
		m_w_gradient.resize(m_perceptron->get_weigh().size());
		m_gradient = 0;
		m_error = 0;
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

		m_gradient = m_gradient*m_step;
	}

	void Steepest_descent_process::compute()
	{
		m_perceptron->set_bia(m_perceptron->get_bia() - m_gradient);

		vector<double> in = get_inputs();
		if (in.size() == m_w_gradient.size())
		{
			vector<double> w = m_perceptron->get_weigh();
			vector<double>::iterator it_w = w.begin();
			for (vector<double>::iterator it = in.begin(); it != in.end(); ++it)
			{
				(*it_w) += (*it)*m_gradient;
				it_w++;
			}

			m_perceptron->set_weigh(w);
		}
		m_gradient = 0;
	}

} // namespace SNN_trainer