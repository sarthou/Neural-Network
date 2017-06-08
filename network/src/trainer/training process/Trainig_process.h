//============================================================================
// Name        : Trainig_process.h
// Authors     : Guillaume Sarthou
// EMail       : open.pode@gmail.com
// Date		   : 8 jun. 2017
// Version     : V1.4
// Copyright   : This file is part of SNN_network project which is released under
//               MIT license.
//============================================================================
#pragma once

#ifndef TRAINING_PROCESS_H
#define TRAINING_PROCESS_H

#include "../../perceptron/Perceptrons.h"
#include <iostream>
#include <vector>

namespace SNN_network
{

	class Trainig_process
	{
	public:
		Trainig_process(Perceptron* p_perceptron);
		virtual ~Trainig_process() {};

		virtual void set_error(double T);
		virtual void propagate(vector<Trainig_process*> process, bool out = false) {};
		virtual void compute();

		virtual void add_to_gradient(double value) { m_gradient += value; };

	protected:
		Perceptron* m_perceptron;

		double m_bia_gradient;
		vector<double> m_w_gradient;

		double m_gradient;
		double m_error;
		double m_delta;

		void derivate_perceptron() { m_perceptron->derivate(); };
		double get_derivate() { return *(m_perceptron->m_derivate.begin()); };
		vector<double> get_inputs();
		void add_to_precedent(vector<Trainig_process*> process, double factor);
	};

} // namespace SNN_network

#endif
