#include "Trainig_process.h"

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
				process[i]->add(factor*m_perceptron->m_w[i]);
		}
	}
}