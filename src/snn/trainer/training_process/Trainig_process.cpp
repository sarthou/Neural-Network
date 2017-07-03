//============================================================================
// Name        : Trainig_process.cpp
// Authors     : Guillaume Sarthou
// EMail       : open.pode@gmail.com
// Date		   : 29 jun. 2017
// Version     : V1.5
// Copyright   : This file is part of SNN_network project which is released under
//               MIT license.
//============================================================================
#include "snn/trainer/training_process/Trainig_process.h"
#include <cmath>

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
		if (1 != m_perceptron->m_out.size())
			cout << "Internal training error" << endl;
		else
			m_error = m_perceptron->m_out[0] - T;
	}

	void Trainig_process::compute()
	{
		m_perceptron->m_bia -= m_delta;

		if (m_perceptron->m_in.size() == m_w_gradient.size())
		{
			for (unsigned int i = 0; i < m_perceptron->m_in.size(); i++)
				m_perceptron->m_w[i] += (*m_perceptron->m_in[i])[0] * m_delta;
		}
		m_gradient = 0;
	}

	void Trainig_process::add_to_precedent(vector<Trainig_process*>* process, double factor)
	{
		if(m_perceptron->m_input_perceptrons != nullptr)
			if (m_perceptron->m_input_perceptrons->size())
			{
				if (m_perceptron->m_w.size() == process->size())
				{
					for (unsigned int i = 0; i < process->size(); i++)
						(*process)[i]->add_to_gradient(factor*m_perceptron->m_w[i]);
				}
			}
	}

} // namespace SNN_network