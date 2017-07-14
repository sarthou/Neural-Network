//============================================================================
// Name        : Perceptron_input.h
// Authors     : Guillaume Sarthou
// EMail       : open.pode@gmail.com
// Date		   : 4 jul. 2017
// Version     : V2.0
// Copyright   : This file is part of SNN_network project which is released under
//               MIT license.
//============================================================================

#include "snn/perceptron/Perceptron.h"

#ifndef PERCEPTRON_INPUT_H
#define PERCEPTRON_INPUT_H

namespace SNN
{
	class Perceptron_input : public Perceptron
	{
	public:
		Perceptron_input(int p_layer, int p_id) : Perceptron(p_layer, p_id, input) {};
		Perceptron_input(Perceptron_input const& perceptron) : Perceptron(perceptron) {};
		~Perceptron_input() {};

		Perceptron_input& operator=(Perceptron_input const& perceptron) { Perceptron::operator=(perceptron); return *this; };

		inline void set_input(const vector<float>& p_input) { m_out = p_input; };

	};
} // namespace SNN_network

#endif