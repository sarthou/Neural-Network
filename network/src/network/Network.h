#pragma once

#ifndef NETWORK_H
#define NETWORK_H

#include "../perceptron/Perceptron.h"
#include "../perceptron/Perceptrons.h"
#include <vector>
#include <string>
#include <iostream>

namespace SNN_network
{
	class Trainer;

	enum perceptron_type_t
	{
		identities,
		binary_step,
		logistic,
		tanH,
		arcTan,
		softsign,
		rectifier,
		rectifier_param,
		ELU,
		softPlus,
		bent_identity,
		sinusoid,
		sinc,
		gaussian
	};

	using namespace std;
	class Network
	{
		friend class Trainer;
	public:
		Network();
		Network(vector<int> p_nb_perceptrons);
		Network(vector<perceptron_type_t> p_types);
		Network(vector<perceptron_type_t> p_types, vector<double> p_params);
		Network(vector<int> p_nb_perceptrons, vector<perceptron_type_t> p_types);
		Network(vector<int> p_nb_perceptrons, vector<perceptron_type_t> p_types, vector<double> p_params);
		~Network();

		void print();
		void print_output();
		void sim(vector<vector<double>*> P, bool clr = true);
		void clr_internal_values();

		vector<vector<double>>* get_output() { return &m_out; };
		vector<vector<double>> get_output_cpy() { return m_out; };

		void round_output();

	private:
		vector<vector<double>> m_out;
		vector<vector<Perceptron*>> m_perceptrons;
		vector<int> m_nb_perceptrons;
		vector<perceptron_type_t> m_types;
		vector<double> m_params;
		bool m_is_train;
		bool m_is_configure;

		bool is_configure() { return m_is_configure; };
		void set_it_train() { if (m_is_configure) m_is_train = true; };

		void generate_network();
		void link_network();

		void init();
		void faile_to_configure();
		bool vector_is_uniforme(vector<vector<double>*>& p_vector);

		Perceptron* creat_perceptron(int layer, int id, perceptron_type_t type, double param = 0);
	};

} // namespace SNN_network

#endif
