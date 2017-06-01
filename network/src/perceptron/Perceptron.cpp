#include "Perceptron.h"

Perceptron::Perceptron(int p_layer, int p_id)
{
	m_layer = p_layer;
	m_id = p_id;
	m_input_perceptrons = nullptr;
}

Perceptron::~Perceptron()
{
	if (m_input_perceptrons)
		if(m_input_perceptrons->size() == 0)
			delete m_input_perceptrons;
}

void Perceptron::set_input(vector<Perceptron*>* p_input_perceptrons)
{
	m_input_perceptrons = p_input_perceptrons;
	for (vector<Perceptron*>::iterator it = m_input_perceptrons->begin(); it != m_input_perceptrons->end(); ++it)
	{
		m_in.push_back((*it)->get_output());
	}
	m_w.resize(m_input_perceptrons->size());
}

bool Perceptron::set_input(vector<vector<double>*> p_input)
{
	bool ok = true;
	m_input_perceptrons = new vector<Perceptron*>;

	if (m_w.size() == 0)
		m_w.resize(p_input.size());
	else if (m_w.size() != p_input.size())
		ok = false;

	if(ok)
		m_in = p_input;

	return ok;
}

void Perceptron::set_weigh(vector<double> p_w)
{
	if (p_w.size() == m_w.size())
		m_w = p_w;
}

void Perceptron::sum()
{
	vector<double> tmp_sum;
	if (m_w.size() > 0)
	{
		int size = m_in.at(0)->size();
		tmp_sum.resize(size);
		for (vector<double>::iterator it = tmp_sum.begin(); it != tmp_sum.end(); ++it)
			(*it) = -m_bia;

		vector<double>::iterator it_w = m_w.begin();
		for (vector<vector<double>*>::iterator it_vect = m_in.begin(); it_vect != m_in.end(); ++it_vect)
		{
			vector<double>::iterator it_out = tmp_sum.begin();
			for (vector<double>::iterator it_in = (*it_vect)->begin(); it_in != (*it_vect)->end(); ++it_in)
			{
				(*it_out) += (*it_w)*(*it_in);
				++it_out;
			}
			++it_w;
		}
	}
	m_sum = tmp_sum;
}