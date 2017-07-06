//============================================================================
// Name        : Perceptron.h
// Authors     : Guillaume Sarthou
// EMail       : open.pode@gmail.com
// Date		   : 29 jun. 2017
// Version     : V1.5
// Copyright   : This file is part of SNN_network project which is released under
//               MIT license.
//============================================================================

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
		virtual bool set_input(const vector<vector<float>*>& p_input);
		virtual bool set_input(const vector<float>& p_input) { p_input; return true; };

		void set_weigh(const vector<float>& p_w);
		vector<float> get_weigh() { return m_w; };
		void set_bia(float p_bia) { m_bia = p_bia; };
		float get_bia() { return m_bia; };

		vector<float>* get_output() { return &m_out; };
		vector<float> get_output_cpy() { return m_out; };
		void clr() { m_out.clear(); m_sum.clear(); m_derivate.clear(); };

		virtual void activate() = 0;
		virtual string get_type() = 0;

	protected:
		int m_layer;
		int m_id;

		float m_bia;
		vector<float> m_w;

		vector<Perceptron*>* m_input_perceptrons;
		vector<float> m_out;

		vector<float> m_sum;
		vector<float> m_derivate;

		void sum();
		virtual void derivate() = 0;
		virtual float derivate_single() = 0;
	};

} // namespace SNN_network

#endif