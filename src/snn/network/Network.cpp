//============================================================================
// Name        : Network.cpp
// Authors     : Guillaume Sarthou
// EMail       : open.pode@gmail.com
// Date		   : 18 jul. 2017
// Version     : V2.0
// Copyright   : This file is part of SNN_network project which is released under
//               MIT license.
//============================================================================
#include "snn/network/Network.h"
#include <math.h>
#include <vector>

#ifdef _WIN32
#include <Windows.h>
#endif

using namespace std;

namespace SNN
{

	Network::Network()
	{
		m_is_train = false;
		generate_network();
		link_network();
		m_is_configure = true;
	}

	Network::Network(vector<int>& p_nb_perceptrons)
	{
		if (!vector_is_positive(p_nb_perceptrons))
			cout << "non positive numbers of perceptrons" << endl;
		else
		{
			m_is_train = false;
			m_nb_perceptrons = p_nb_perceptrons;
			generate_network();
			link_network();
			m_is_configure = true;
		}
		faile_to_configure();
	}

	Network::Network(vector<perceptron_type_t>& p_types)
	{
		init();
		if (p_types.size() > 1)
			cout << "Too many types" << endl;
		else
		{
			m_types = p_types;
			generate_network();
			link_network();
			m_is_configure = true;
		}
		faile_to_configure();
	}

	Network::Network(vector<perceptron_type_t>& p_types, vector<float>& p_params)
	{
		init();
		if (p_types.size() > 1)
			cout << "Too many types" << endl;
		else if (p_params.size() > 1)
			cout << "Too many parameters" << endl;
		else
		{
			m_types = p_types;
			m_params = p_params;
			generate_network();
			link_network();
			m_is_configure = true;
		}
		faile_to_configure();
	}

	Network::Network(vector<int>& p_nb_perceptrons, vector<perceptron_type_t>& p_types)
	{
		init();
		if (p_types.size() > p_nb_perceptrons.size() + 1)
			cout << "Too many types" << endl;
		else if ((p_types.size() > 2) && (p_types.size() != p_nb_perceptrons.size() + 1))
		{
			if (p_types.size() == p_nb_perceptrons.size())
				cout << "Miss type of output perceptrons" << endl;
			else
				cout << "Incompatible size of type's vector" << endl;
		}
		else
		{
			if (!vector_is_positive(p_nb_perceptrons))
				cout << "non positive numbers of perceptrons" << endl;
			else
			{
				m_nb_perceptrons = p_nb_perceptrons;
				m_types = p_types;
				generate_network();
				link_network();
				m_is_configure = true;
			}
		}
		faile_to_configure();
	}

	Network::Network(vector<int>& p_nb_perceptrons, vector<perceptron_type_t>& p_types, vector<float>& p_params)
	{
		init();
		if (p_types.size() > p_nb_perceptrons.size() + 1)
			cout << "Too many types" << endl;
		else if ((p_types.size() > 2) && (p_types.size() != p_nb_perceptrons.size() + 1))
		{
			if (p_types.size() == p_nb_perceptrons.size())
				cout << "Miss type of output perceptrons" << endl;
			else
				cout << "Incompatible size of type's vector" << endl;
		}
		else if (p_params.size() > p_types.size())
			cout << "Too many parameters" << endl;
		else if ((p_params.size() > 2) && (p_types.size() != p_params.size()))
			cout << "Incompatible size of parameter's vector" << endl;
		else
		{
			if (!vector_is_positive(p_nb_perceptrons))
				cout << "non positive numbers of perceptrons" << endl;
			else
			{
				m_nb_perceptrons = p_nb_perceptrons;
				m_types = p_types;
				m_params = p_params;
				generate_network();
				link_network();
				m_is_configure = true;
			}
		}
		faile_to_configure();
	}

	Network::Network(Network const& network)
	{
		m_out = network.m_out;
		m_perceptrons = vector<vector<Perceptron*> >();
		m_nb_perceptrons = network.m_nb_perceptrons;
		m_types = network.m_types;
		m_params = network.m_params;
		m_is_train = network.m_is_train;
		m_is_configure = network.m_is_configure;

		generate_copy_network(network);
		link_network_copy();
	}

	Network::~Network()
	{
		for (vector<vector<Perceptron*> >::iterator it_layer = m_perceptrons.begin(); it_layer != m_perceptrons.end(); ++it_layer)
		{
			for (vector<Perceptron*>::iterator it_perceptron = it_layer->begin(); it_perceptron != it_layer->end(); ++it_perceptron)
			{
				delete (*it_perceptron);
				*it_perceptron = nullptr;
			}
		}

		for (unsigned int i = 0; i < m_P.size(); i++)
			delete(m_P[i]);
	}

	Network& Network::operator=(Network const& network)
	{
		if (this != &network)
		{
			m_out = network.m_out;
			m_perceptrons = vector<vector<Perceptron*> >();
			m_nb_perceptrons = network.m_nb_perceptrons;
			m_types = network.m_types;
			m_params = network.m_params;
			m_is_train = network.m_is_train;
			m_is_configure = network.m_is_configure;

			generate_copy_network(network);
			link_network_copy();
		}
		return *this;
	}

	void Network::print()
	{
		cout << "Configure: ";
		if (m_is_configure)
			cout << "true" << endl;
		else
			cout << "false" << endl;

		cout << "Train \t : ";
		if (m_is_train)
			cout << "true" << endl;
		else
			cout << "false" << endl;

		cout << endl;

		if (m_is_configure)
		{
			if (m_perceptrons[0].size() == 0)
			{
#ifdef _WIN32
				SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), 0x01);
#endif
				cout << "Input not gererated yet" << endl;
#ifdef _WIN32
				SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), 0x07);
#endif
			}

			int layer = 0;
			for (vector<vector<Perceptron*> >::iterator it_layer = m_perceptrons.begin(); it_layer != m_perceptrons.end(); ++it_layer)
			{
				if ((*it_layer).size() != 0)
				{
					vector<Perceptron*>::iterator it_perceptron = it_layer->begin();
					string type = ((*it_perceptron)->get_type)();
					int nb_percep = (*it_layer).size();

					cout << "Layer " << layer << "\t:\t" << nb_percep << "\tperceptrons " << type << endl;

					layer++;
				}
			}

			if (m_perceptrons[m_nb_perceptrons.size() + 1].size() == 0)
			{
#ifdef _WIN32
				SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), 0x01);
#endif
				cout << "Output not gererated yet" << endl;
#ifdef _WIN32
				SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), 0x07);
#endif
			}
		}
	}

	void Network::print_output()
	{
		if (m_out.size() > 0)
		{
			for (vector<vector<float> >::iterator it = m_out.begin(); it != m_out.end(); ++it)
			{
				for (vector<float>::iterator it2 = it->begin(); it2 != it->end(); ++it2)
				{
					std::cout << ' ' << *it2;
				}
				std::cout << '\n';
			}
			std::cout << '\n';
		}
		else
		{
#ifdef _WIN32
			SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), 0x0E);
#endif
			cout << "no output" << endl;
#ifdef _WIN32
			SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), 0x07);
#endif
		}
	}

	void Network::sim(vector<vector<float> >* P, bool clr)
	{
		Matrix<float> mat(P->size(), (*P)[0].size(), *P);
		sim(mat, clr);
	}

	void Network::sim(Matrix<float>&  P, bool clr)
	{
#ifdef _WIN32
		SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), 0x0C);
#endif
		if (m_is_train)
		{
			m_out.clear();

			//set data into input layer
			for (unsigned int id = 0; id < m_perceptrons[0].size(); id++)
				m_perceptrons[0][id]->set_input(vector<float>(P.get_row(id), P.get_row(id)+P.get_col_count()));

			unsigned int last_layer = m_perceptrons.size() - 1;

			//activate internal layers
			for (unsigned int layer = 1; layer < last_layer; layer++)
				for (unsigned int id = 0; id < m_perceptrons[layer].size(); id++)
					m_perceptrons[layer][id]->activate();

			//activate output layer
			for (unsigned int id = 0; id < m_perceptrons[last_layer].size(); id++)
			{
				(m_perceptrons[last_layer][id]->activate)();
				m_out.push_back(m_perceptrons[last_layer][id]->get_output_cpy());
			}

			if (clr)
				clr_internal_values();
		}
		else
			cout << "Sim => Network not train." << endl;
#ifdef _WIN32
		SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), 0x07);
#endif
	}

	void Network::clr_internal_values()
	{
		for (unsigned int layer = 0; layer < m_perceptrons.size(); layer++)
			for (unsigned int percept = 0; percept < m_perceptrons[layer].size(); percept++)
				m_perceptrons[layer][percept]->clr();
	}

	void Network::generate_network()
	{
		int layer = 0;
		m_perceptrons.push_back(vector<Perceptron*>()); // insert empty vector pour input
		for (vector<int>::iterator it = m_nb_perceptrons.begin(); it != m_nb_perceptrons.end(); ++it)
		{
			vector<Perceptron*> temp_vect;

			perceptron_type_t type = logistic;
			if (m_types.size() > 0)
			{
				if (m_types.size() <= 2)
					type = m_types[0];
				else
					type = m_types[layer];
			}

			float param = 0;
			if (m_params.size() > 0)
			{
				if (m_params.size() <= 2)
					param = m_params[0];
				else
					param = m_params[layer];
			}

			for (int i = 0; i < (*it); i++)
			{
				temp_vect.push_back(creat_perceptron(layer, i, type, param));
			}
			m_perceptrons.push_back(temp_vect);
			layer++;
		}
		m_perceptrons.push_back(vector<Perceptron*>()); // insert empty vector pour output
	}

	void Network::generate_copy_network(Network const& network)
	{
		int layer = 0;
		for (vector<vector<Perceptron*> >::const_iterator it = network.m_perceptrons.begin(); it != network.m_perceptrons.end(); ++it)
		{
			vector<Perceptron*> temp_vect;

			for (unsigned int i = 0; i < (*it).size(); i++)
			{
				temp_vect.push_back(copy_perceptron(*(network.m_perceptrons[layer][i])));
			}
			m_perceptrons.push_back(temp_vect);
			layer++;
		}
	}

	void Network::link_network()
	{
		if (m_nb_perceptrons.size() > 1)
		{
			vector<vector<Perceptron*> >::iterator init_it = m_perceptrons.begin() + 1; // don't link with input layer
			for (vector<vector<Perceptron*> >::iterator it = init_it + 1; it != m_perceptrons.end() - 1; ++it)
			{
				for (vector<Perceptron*>::iterator percept_it = it->begin(); percept_it != it->end(); percept_it++)
				{
					(*percept_it)->set_input(&(*init_it));
				}
				init_it++;
			}
		}
	}

	void Network::link_network_copy()
	{
		if (m_nb_perceptrons.size() > 0)
		{
			vector<vector<Perceptron*> >::iterator init_it = m_perceptrons.begin();
			for (vector<vector<Perceptron*> >::iterator it = init_it + 1; it != m_perceptrons.end(); ++it)
			{
				for (vector<Perceptron*>::iterator percept_it = it->begin(); percept_it != it->end(); percept_it++)
				{
					(*percept_it)->set_input(&(*init_it));
				}
				init_it++;
			}
		}
	}

	void Network::init()
	{
#ifdef _WIN32
		SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), 0x0E);
#endif
		m_is_train = false;
		m_is_configure = false;
	}

	void Network::faile_to_configure()
	{
		if (!m_is_configure)
		{
#ifdef WINDOWS
			SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), 0x0C);
#endif
			cout << "Network => Fail to configure the network" << endl;
		}
#ifdef WINDOWS
		SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), 0x07);
#endif
	}

	bool Network::vector_is_uniforme(vector<vector<float>*>& p_vector)
	{
		bool uniform = true;
		vector<vector<float>*>::iterator it_begin = p_vector.begin();
		unsigned int size = (*it_begin)->size();
		for (vector<vector<float>*>::iterator it = it_begin + 1; it != p_vector.end(); ++it)
		{
			if (size != (*it)->size())
				uniform = false;
		}

		if (p_vector.size() == 0)
			uniform = false;

		return uniform;
	}


	bool Network::vector_is_positive(vector<int>& p_vector)
	{
		bool positive = true;
		for (vector<int>::iterator it = p_vector.begin(); it != p_vector.end(); ++it)
		{
			if ((*it) <= 0)
				positive = false;
		}
		return positive;
	}

	Perceptron* Network::creat_perceptron(int layer, int id, perceptron_type_t type, float param)
	{
		Perceptron* tmp_perceptron = new Perceptron(layer, id, type, param);
		return tmp_perceptron;
	}

	Perceptron* Network::copy_perceptron(Perceptron& perceptron)
	{
		Perceptron* tmp_perceptron = nullptr;
		if( (perceptron.get_type)() == "input")
			tmp_perceptron = new Perceptron_input((Perceptron_input&)perceptron);
		else 
			tmp_perceptron = new Perceptron((Perceptron&)perceptron);

		return tmp_perceptron;
	}

	void Network::round_output()
	{
		for (vector<vector<float> >::iterator it_vect = m_out.begin(); it_vect != m_out.end(); ++it_vect)
			for (vector<float>::iterator it = it_vect->begin(); it != it_vect->end(); ++it)
				(*it) = round(*it);
	}

} // namespace SNN_network