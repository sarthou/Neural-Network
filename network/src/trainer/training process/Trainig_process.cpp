//============================================================================
// Name        : Trainig_process.cpp
// Authors     : Guillaume Sarthou
// EMail       : open.pode@gmail.com
// Date		   : 25 jun. 2017
// Version     : V1.2
// Copyright   : This file is part of SNN_network project which is released under
//               MIT license.
//============================================================================
#include "Trainig_process.h"

namespace SNN
{
	Trainig_process::Trainig_process(Perceptron* p_perceptron) 
	{ 
		m_perceptron = p_perceptron; 
		m_delta = 0;
		m_error = 0;
		m_gradient = 0;
		m_w_gradient.resize(m_perceptron->get_weigh().size());
	}

	vector<double> Trainig_process::get_inputs()
	{
		vector<double> tmp;
		for (vector<vector<double>*>::iterator it = m_perceptron->m_in.begin(); it != m_perceptron->m_in.end(); ++it)
		{
			tmp.push_back(*((*it)->begin()));
		}
		return tmp;
	}

	void Trainig_process::set_error(double T)
	{
		if (1 != m_perceptron->get_output()->size())
			cout << "Internal training error" << endl;
		else
		{
			vector<double>::iterator out_it = m_perceptron->get_output()->begin();
			m_error = (*out_it) - T;
		}
	}

	void Trainig_process::compute()
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
		m_gradient = 0;
	}

	void Trainig_process::add_to_precedent(vector<Trainig_process*> process, double factor)
	{
		if (m_perceptron->m_input_perceptrons->size())
		{
			if (m_perceptron->m_w.size() == process.size())
			{
				for (unsigned int i = 0; i < process.size(); i++)
					process[i]->add_to_gradient(factor*m_perceptron->m_w[i]);
			}
		}
	}

} // namespace SNN_network