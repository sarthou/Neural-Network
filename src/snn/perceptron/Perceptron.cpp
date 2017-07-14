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
#include "snn/perceptron/Perceptrons.h"

#include <cmath>

namespace SNN
{

	Perceptron::Perceptron(int p_layer, int p_id, perceptron_type_t type, float param)
	{
		m_layer = p_layer;
		m_id = p_id;
		m_type = type;
		m_a = param;
		m_input_perceptrons = nullptr;

		set_functions();
	}

	Perceptron::Perceptron(Perceptron const& perceptron)
	{
		if (this != &perceptron)
		{
			m_layer = perceptron.m_layer;
			m_id = perceptron.m_id;

			m_type = perceptron.m_type;
			m_a = perceptron.m_a;

			m_bia = perceptron.m_bia;
			m_w = perceptron.m_w;

			m_input_perceptrons = nullptr;
			m_out = perceptron.m_out;

			m_sum = perceptron.m_sum;
			m_derivate = perceptron.m_derivate;

			set_functions();
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

		m_w.resize(m_input_perceptrons->size());
	}

	bool Perceptron::set_input(const vector<vector<float>*>& p_input)
	{
		p_input;

		return true;
	}

	void Perceptron::set_weigh(const vector<float>& p_w)
	{
		if (p_w.size() == m_w.size())
			m_w = p_w;
	}

	void Perceptron::set_functions()
	{
		switch (m_type)
		{
		case input:
			activate_ptr = &activate_input;
			get_type_ptr = get_type_input;
			derivate_single_ptr = derivate_single_input;
			break;
		case identities:	
			activate_ptr = activate_identities;
			get_type_ptr = get_type_identities;
			derivate_single_ptr = derivate_single_identities;
			break;
		case binary_step:	
			activate_ptr = activate_binary_step;
			get_type_ptr = get_type_binary_step;
			derivate_single_ptr = derivate_single_binary_step;
			break;
		case logistic:		
			activate_ptr = activate_logistic;
			get_type_ptr = get_type_logistic;
			derivate_single_ptr = derivate_single_logistic;
			break;
		case tanH:			
			activate_ptr = activate_tanH;
			get_type_ptr = get_type_tanH;
			derivate_single_ptr = derivate_single_tanH;
			break;
		case arcTan:		
			activate_ptr = activate_arcTan;
			get_type_ptr = get_type_arcTan;
			derivate_single_ptr = derivate_single_arcTan;
			break;
		case softsign:		
			activate_ptr = activate_softsign;
			get_type_ptr = get_type_softsign;
			derivate_single_ptr = derivate_single_softsign;
			break;
		case rectifier:		
			activate_ptr = activate_rectifier;
			get_type_ptr = get_type_rectifier;
			derivate_single_ptr = derivate_single_rectifier;
			break;
		case rectifier_param: 
			activate_ptr = activate_rectifier_param;
			get_type_ptr = get_type_rectifier_param;
			derivate_single_ptr = derivate_single_rectifier_param;
			break;
		case ELU:			
			activate_ptr = activate_ELU;
			get_type_ptr = get_type_ELU;
			derivate_single_ptr = derivate_single_ELU;
			break;
		case softPlus:		
			activate_ptr = activate_softPlus;
			get_type_ptr = get_type_softPlus;
			derivate_single_ptr = derivate_single_softPlus;
			break;
		case bent_identity: 
			activate_ptr = activate_bent_identity;
			get_type_ptr = get_type_bent_identity;
			derivate_single_ptr = derivate_single_bent_identity;
			break;
		case sinusoid:		
			activate_ptr = activate_sinusoid;
			get_type_ptr = get_type_sinusoid;
			derivate_single_ptr = derivate_single_sinusoid;
			break;
		case sinc:			
			activate_ptr = activate_sinc;
			get_type_ptr = get_type_sinc;
			derivate_single_ptr = derivate_single_sinc;
			break;
		case gaussian:		
			activate_ptr = activate_gaussian;
			get_type_ptr = get_type_gaussian;
			derivate_single_ptr = derivate_single_gaussian;
			break;
		default:
			activate_ptr = activate_identities;
			get_type_ptr = get_type_identities;
			derivate_single_ptr = derivate_single_identities;
			break;
		}
	}

} // namespace SNN_network