//============================================================================
// Name        : Serializer.cpp
// Authors     : Guillaume Sarthou
// EMail       : open.pode@gmail.com
// Date		   : 19 jul. 2017
// Version     : V2.1
// Copyright   : This file is part of SNN_network project which is released under
//               MIT license.
//============================================================================

#include <vector>
#include "snn/serializer/Serializer.h"

namespace SNN
{

	vector<char> Serializer::encode(Network& net)
	{
		Serial_vector net_encode;

		/*Globals networks features*/
		net_encode.push_back( (char)net.m_nb_perceptrons.size() ); // nb layers
		for (char i = 0; i < (char)net.m_nb_perceptrons.size(); i++)
			net_encode.push_back((char)net.m_nb_perceptrons[i] ); // nb_perceprtons in each layers

		net_encode.push_back( (char)net.m_types.size() );
		for (char i = 0; i < (char)net.m_types.size(); i++)
			net_encode.push_back( (char)net.m_types[i] ); // types of each layers

		net_encode.push_back( (char)net.m_params.size() );
		for (char i = 0; i < (char)net.m_params.size(); i++)
			net_encode.push_back( net.m_params[i] ); // params of each layers

		/*Input and output description*/
		net_encode.push_back( (unsigned int)net.m_perceptrons.begin()->size() );

		net_encode.push_back( (unsigned int)net.m_perceptrons[net.m_nb_perceptrons.size()].size() );

		/*Perceptrons features*/
		vector<int> nb_percep = net.m_nb_perceptrons;
		nb_percep.push_back(net.m_perceptrons[net.m_nb_perceptrons.size()].size()); // add the output
		for (unsigned int layer = 0; layer < nb_percep.size(); layer++)
		{
			for (int id = 0; id < nb_percep[layer]; id++)
			{
				net_encode.push_back(net.m_perceptrons[layer + 1][id]->get_bia());
				net_encode.push_back((unsigned int)net.m_perceptrons[layer + 1][id]->get_weigh().size());
				for(unsigned int i = 0; i < (unsigned int)net.m_perceptrons[layer + 1][id]->get_weigh().size(); i++)
					net_encode.push_back(net.m_perceptrons[layer + 1][id]->get_weigh()[i]);
			}
		}

		return net_encode.get_data();
	}

	Network Serializer::decode(vector<char> data)
	{
		Serial_vector net_encode(data);
		unsigned int size;

		vector<int> nb;
		size = (size_t)net_encode.get_next_char();
		for (unsigned int i = 0; i < size; i++)
			nb.push_back((int)net_encode.get_next_char());

		vector<perceptron_type_t> type;
		size = (size_t)net_encode.get_next_char();
		for (unsigned int i = 0; i < size; i++)
			type.push_back((perceptron_type_t)net_encode.get_next_char());

		vector<float> param;
		size = (size_t)net_encode.get_next_char();
		for (unsigned int i = 0; i < size; i++)
			param.push_back(net_encode.get_next_float());

		Network net(nb, type, param);

		trainig_config_t config;
		config.nb_epochs = 0;
		Trainer trainer;
		trainer.set_config(config);

		Matrix<float> P_mat(net_encode.get_next_uint(), 1);
		Matrix<float> T_mat(net_encode.get_next_uint(), 1);
		trainer.train(&net, P_mat, T_mat);

		nb.push_back(T_mat.get_row_count());
		for (unsigned int layer = 0; layer < nb.size(); layer++)
		{
			for (int id = 0; id < nb[layer]; id++)
			{
				net.m_perceptrons[layer + 1][id]->set_bia(net_encode.get_next_float());
				vector<float> w;
				w.resize(net_encode.get_next_uint());
				for (unsigned int i = 0; i < w.size(); i++)
					w[i] = net_encode.get_next_float();
				net.m_perceptrons[layer + 1][id]->set_weigh(w);
			}
		}

		net.print();

		return net;
	}
}
