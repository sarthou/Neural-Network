//============================================================================
// Name        : Trainig_process.h
// Authors     : Guillaume Sarthou
// EMail       : open.pode@gmail.com
// Date		   : 29 jun. 2017
// Version     : V1.5
// Copyright   : This file is part of SNN_network project which is released under
//               MIT license.
//============================================================================

#ifndef TRAINING_PROCESS_H
#define TRAINING_PROCESS_H

#include "snn/perceptron/Perceptron.h"
#include "snn/trainer/Training_configuration.h"

#include <iostream>
#include <vector>

namespace SNN
{

	class Trainig_process
	{
	public:
		Trainig_process(Perceptron* p_perceptron);
		virtual ~Trainig_process() {};

		virtual void set_error(float T);
		virtual void propagate(vector<Trainig_process*>* process, bool out = false) { (void)process; (void)out; };
		virtual void compute();

		inline void add_to_gradient(float value) { m_gradient += value; };

	protected:
		Perceptron* m_perceptron;

		float m_bia_gradient;
		vector<float> m_w_gradient;

		float m_gradient;
		float m_error;
		float m_delta;

		inline float get_single_derivate() { return m_perceptron->derivate_single(); };

		void add_to_precedent(vector<Trainig_process*>* process, float factor);
	};

} // namespace SNN_network

#endif
