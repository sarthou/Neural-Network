//============================================================================
// Name        : Trainig_process.cpp
// Authors     : Guillaume Sarthou
// EMail       : open.pode@gmail.com
// Date		   : 18 jul. 2017
// Version     : V2.0
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

	void Trainig_process::set_error(float T)
	{
		m_error = m_perceptron->m_out[0] - T;
	}

	void Trainig_process::compute()
	{
		m_perceptron->m_bia -= m_delta;

		vector<Perceptron*>* percpts = m_perceptron->m_input_perceptrons;
		vector<float>* w = &m_perceptron->m_w;
		unsigned int size = percpts->size();
 		for (unsigned int i = 0; i < size; i++)
			(*w)[i] += (*percpts)[i]->m_out[0] * m_delta;

		m_gradient = 0;
	}

	void Trainig_process::add_to_precedent(vector<Trainig_process*>* process, float factor)
	{
		if (m_perceptron->m_layer > 0)
		{
			unsigned int size = process->size();
			for (unsigned int i = 0; i < size; i++)
				(*process)[i]->add_to_gradient(factor*m_perceptron->m_w[i]);
		}
	}

} // namespace SNN_network