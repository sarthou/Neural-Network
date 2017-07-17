//============================================================================
// Name        : Network.h
// Authors     : Guillaume Sarthou
// EMail       : open.pode@gmail.com
// Date		   : 18 jul. 2017
// Version     : V2.0
// Copyright   : This file is part of SNN_network project which is released under
//               MIT license.
//============================================================================

#ifndef NETWORK_H
#define NETWORK_H

#include "snn/perceptron/Perceptron.h"
#include "snn/perceptron/Perceptron_input.h"

#include "snn/utility/Matrix.h"
#include <vector>
#include <string>
#include <iostream>

namespace SNN
{
	class Trainer;

	using namespace std;
	class Network
	{
		friend class Trainer;
	public:
		Network();
		Network(vector<int> p_nb_perceptrons);
		Network(vector<perceptron_type_t> p_types);
		Network(vector<perceptron_type_t> p_types, vector<float> p_params);
		Network(vector<int> p_nb_perceptrons, vector<perceptron_type_t> p_types);
		Network(vector<int> p_nb_perceptrons, vector<perceptron_type_t> p_types, vector<float> p_params);
		Network(Network const& network);
		~Network();

		Network& operator=(Network const& network);

		void print();
		void print_output();
		void sim(vector<vector<float> >* P, bool clr = true);
		void clr_internal_values();

		inline vector<vector<float> >* get_output() { return &m_out; };
		inline vector<vector<float> > get_output_cpy() { return m_out; };

		void round_output();

	private:
		vector<vector<float>*> m_P;
		vector<vector<float>> m_out;
		vector<vector<Perceptron*>> m_perceptrons;
		vector<int> m_nb_perceptrons;
		vector<perceptron_type_t> m_types;
		vector<float> m_params;
		bool m_is_train;
		bool m_is_configure;

		bool is_configure() { return m_is_configure; };
		void set_it_train() { if (m_is_configure) m_is_train = true; };

		void sim(Matrix<float>&  P, bool clr = true);

		void generate_network();
		void generate_copy_network(Network const& network);
		void link_network();
		void link_network_copy();

		void init();
		void faile_to_configure();
		bool vector_is_uniforme(vector<vector<float>*>& p_vector);
		bool vector_is_positive(vector<int>& p_vector);

		Perceptron* creat_perceptron(int layer, int id, perceptron_type_t type, float param = 0);
		Perceptron* copy_perceptron(Perceptron& perceptron);
	};

} // namespace SNN_network

#endif