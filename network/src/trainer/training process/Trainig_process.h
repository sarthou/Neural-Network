//============================================================================
// Name        : Trainig_process.h
// Authors     : Guillaume Sarthou
// EMail       : open.pode@gmail.com
// Date		   : 6 jun. 2017
// Version     : V1.2
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
		Trainig_process(Perceptron* p_perceptron) { m_perceptron = p_perceptron; };
		virtual ~Trainig_process() {};

		virtual void init() = 0;
		virtual void set_error(double T) {};
		virtual void propagate(vector<Trainig_process*> process, bool out = false) {};
		virtual void compute() {};

		virtual void add(double value) {};

	protected:
		Perceptron* m_perceptron;

		void derivate_perceptron() { m_perceptron->derivate(); };
		double get_derivate() { return *(m_perceptron->m_derivate.begin()); };
		vector<double> get_inputs();
		void add_to_precedent(vector<Trainig_process*> process, double factor);
	};

} // namespace SNN_network

#endif