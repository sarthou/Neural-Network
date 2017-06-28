//============================================================================
// Name        : Perceptron.h
// Authors     : Guillaume Sarthou
// EMail       : open.pode@gmail.com
// Date		   : 25 jun. 2017
// Version     : V1.4
// Copyright   : This file is part of SNN_network project which is released under
//               MIT license.
//============================================================================

#pragma once

#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include <vector>
#include <string>

namespace SNN
{
	using namespace std;

	class Trainig_process;

	class Perceptron
	{
		friend class Trainig_process;
		friend class Trainer;
	public:
		Perceptron(int p_layer, int p_id);
		Perceptron(Perceptron const& perceptron);
		virtual ~Perceptron();

		Perceptron& operator=(Perceptron const& perceptron);

		void set_input(vector<Perceptron*>* p_input_perceptrons);
		bool set_input(vector<vector<double>*> p_input);

		void set_weigh(vector<double> p_w);
		vector<double> get_weigh() { return m_w; };
		void set_bia(double p_bia) { m_bia = p_bia; };
		double get_bia() { return m_bia; };

		vector<double>* get_output() { return &m_out; };
		vector<double> get_output_cpy() { return m_out; };
		void clr() { m_out.clear(); m_sum.clear(); m_derivate.clear(); };

		virtual void activate() = 0;
		virtual string get_type() = 0;

	protected:
		int m_layer;
		int m_id;

		double m_bia;
		vector<double> m_w;

		vector<Perceptron*>* m_input_perceptrons;
		vector<vector<double>*> m_in;
		vector<double> m_out;

		vector<double> m_sum;
		vector<double> m_derivate;

		void sum();
		virtual void derivate() = 0;
	};

} // namespace SNN_network

#endif
