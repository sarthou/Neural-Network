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
		m_delta_1 = 0;
		m_delta = 0;
		m_error = 0;
		m_gradient = 0;
		ss_gradient = 0;
	}

	GD_adagrad_process::~GD_adagrad_process()
	{

	}

	void GD_adagrad_process::init()
	{
		m_w_gradient.resize(m_perceptron->get_weigh().size());
		m_delta_1 = m_delta;
		m_delta = 0;
		m_error = 0;
	}

	void GD_adagrad_process::set_error(double T)
	{
		if (1 != m_perceptron->get_output()->size())
			cout << "Internal training error" << endl;
		else
		{
			vector<double>::iterator out_it = m_perceptron->get_output()->begin();
			m_error = (*out_it) - T;
		}
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

	void GD_adagrad_process::compute()
	{
		m_perceptron->set_bia(m_perceptron->get_bia() - m_delta);

		vector<double> in = get_inputs();
		if (in.size() == m_w_gradient.size())
		{
			vector<double> w = m_perceptron->get_weigh();
			vector<double>::iterator it_w = w.begin();
			for (vector<double>::iterator it = in.begin(); it != in.end(); ++it)
			{
				(*it_w) += (*it)*m_delta;
				it_w++;
			}

			m_perceptron->set_weigh(w);
		}
	}

} // namespace SNN_trainer