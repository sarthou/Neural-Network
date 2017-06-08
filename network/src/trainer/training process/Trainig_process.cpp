//============================================================================
// Name        : Trainig_process.cpp
// Authors     : Guillaume Sarthou
// EMail       : open.pode@gmail.com
// Date		   : 6 jun. 2017
// Version     : V1.2
// Copyright   : This file is part of SNN_network project which is released under
//               MIT license.
//============================================================================
#include "Trainig_process.h"

namespace SNN_network
{

	using namespace SNN_network;

	vector<double> Trainig_process::get_inputs()
	{
		vector<double> tmp;
		for (vector<vector<double>*>::iterator it = m_perceptron->m_in.begin(); it != m_perceptron->m_in.end(); ++it)
		{
			tmp.push_back(*((*it)->begin()));
		}
		return tmp;
	}

	void Trainig_process::add_to_precedent(vector<Trainig_process*> process, double factor)
	{
		if (m_perceptron->m_input_perceptrons->size())
		{
			if (m_perceptron->m_w.size() == process.size())
			{
				for (int i = 0; i < process.size(); i++)
					process[i]->add_to_gradient(factor*m_perceptron->m_w[i]);
			}
		}
	}

} // namespace SNN_network