//============================================================================
// Name        : Steepest_descent_process.cpp
// Authors     : Guillaume Sarthou
// EMail       : open.pode@gmail.com
// Date		   : 13 apr. 2017
// Version     : V1.0
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

	void Steepest_descent_process::init()
	{
		m_w_gradient.resize(m_perceptron->get_weigh().size());
		m_delta = 0;
		m_error = 0;
	}

	void Steepest_descent_process::set_error(double T)
	{
		if (1 != m_perceptron->get_output()->size())
			cout << "Internal training error" << endl;
		else
		{
			vector<double>::iterator out_it = m_perceptron->get_output()->begin();
			m_error = (*out_it) - T;
		}
	}

	void Steepest_descent_process::propagate(vector<Trainig_process*> process, bool out)
	{
		derivate_perceptron();

		if (out)
			m_delta = -m_error*get_derivate();
		else
			m_delta *= get_derivate();

		add_to_precedent(process, m_delta);
	}

	void Steepest_descent_process::compute()
	{
		m_bia_gradient = m_delta;
		m_perceptron->set_bia(m_perceptron->get_bia() - m_bia_gradient*m_step);

		vector<double> in = get_inputs();
		if (in.size() == m_w_gradient.size())
		{
			vector<double> w = m_perceptron->get_weigh();
			vector<double>::iterator it_w = w.begin();
			for (vector<double>::iterator it = in.begin(); it != in.end(); ++it)
			{
				(*it_w) += m_step*(*it)*m_delta;
				it_w++;
			}

			m_perceptron->set_weigh(w);
		}
	}

	void Steepest_descent_process::add(double value)
	{
		m_delta += value;
	}

} // namespace SNN_trainer