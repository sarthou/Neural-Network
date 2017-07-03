//============================================================================
// Name        : Perceptron.cpp
// Authors     : Guillaume Sarthou
// EMail       : open.pode@gmail.com
// Date		   : 29 jun. 2017
// Version     : V1.5
// Copyright   : This file is part of SNN_network project which is released under
//               MIT license.
//============================================================================
#include "snn/perceptron/Perceptron.h"

#include <cmath>

namespace SNN
{

	Perceptron::Perceptron(int p_layer, int p_id)
	{
		m_layer = p_layer;
		m_id = p_id;
		m_input_perceptrons = nullptr;
	}

	Perceptron::Perceptron(Perceptron const& perceptron)
	{
		if (this != &perceptron)
		{
			m_layer = perceptron.m_layer;
			m_id = perceptron.m_id;

			m_bia = perceptron.m_bia;
			m_w = perceptron.m_w;

			m_input_perceptrons = nullptr;
			m_out = perceptron.m_out;

			m_sum = perceptron.m_sum;
			m_derivate = perceptron.m_derivate;
		}
	}

	Perceptron::~Perceptron()
	{
		if (m_input_perceptrons)
			if (m_input_perceptrons->size() == 0)
				delete m_input_perceptrons;
	}

	Perceptron& Perceptron::operator=(Perceptron const& perceptron)
	{
		if (this != &perceptron)
		{
			m_layer = perceptron.m_layer;
			m_id = perceptron.m_id;

			m_bia = perceptron.m_bia;
			m_w = perceptron.m_w;

			if (m_input_perceptrons)
				if (m_input_perceptrons->size() == 0)
					delete m_input_perceptrons;
			m_input_perceptrons = nullptr;
			m_out = perceptron.m_out;

			m_sum = perceptron.m_sum;
			m_derivate = perceptron.m_derivate;
		}
		return *this;
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

	bool Perceptron::set_input(const vector<vector<double>*>& p_input)
	{
		bool ok = true;
		m_input_perceptrons = nullptr;

		if (m_w.size() == 0)
			m_w.resize(p_input.size());
		else if (m_w.size() != p_input.size())
			ok = false;

		if (ok)
			m_in = p_input;

		return ok;
	}

	void Perceptron::set_weigh(const vector<double>& p_w)
	{
		if (p_w.size() == m_w.size())
			m_w = p_w;
	}

	void Perceptron::sum()
	{
		if (m_w.size() > 0)
		{
			m_sum.resize(m_in.at(0)->size());
			for (unsigned int i = 0; i < m_sum.size(); i++)
				m_sum[i] = -m_bia;

			for (unsigned int vect = 0; vect < m_in.size(); vect++)
			{
				for (unsigned int i = 0; i < m_in[vect]->size(); i++)
					m_sum[i] += m_w[vect] * (*m_in[vect])[i];
			}
		}
	}

} // namespace SNN_network