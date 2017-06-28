//============================================================================
// Name        : Network.cpp
// Authors     : Guillaume Sarthou
// EMail       : open.pode@gmail.com
// Date		   : 25 jun. 2017
// Version     : V1.4
// Copyright   : This file is part of SNN_network project which is released under
//               MIT license.
//============================================================================

#include "Network.h"
#include <windows.h>
#include <math.h>

namespace SNN
{

	Network::Network()
	{
		m_is_train = false;
		generate_network();
		link_network();
		m_is_configure = true;
	}

	Network::Network(vector<int> p_nb_perceptrons)
	{
		if(!vector_is_positive(p_nb_perceptrons))
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

	Network::Network(vector<perceptron_type_t> p_types)
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

	Network::Network(vector<perceptron_type_t> p_types, vector<double> p_params)
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

	Network::Network(vector<int> p_nb_perceptrons, vector<perceptron_type_t> p_types)
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

	Network::Network(vector<int> p_nb_perceptrons, vector<perceptron_type_t> p_types, vector<double> p_params)
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
		vector<vector<Perceptron*> > m_perceptrons;
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
	}

	Network& Network::operator=(Network const& network)
	{
		if (this != &network)
		{
			m_out = network.m_out;
			vector<vector<Perceptron*> > m_perceptrons;
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
			int layer = 0;
			for (vector<vector<Perceptron*> >::iterator it_layer = m_perceptrons.begin(); it_layer != m_perceptrons.end(); ++it_layer)
			{
				vector<Perceptron*>::iterator it_perceptron = it_layer->begin();
				string type = (*it_perceptron)->get_type();
				int nb_percep = (*it_layer).size();

				cout << "Layer " << layer << "\t:\t" << nb_percep << "\tperceptrons " << type << endl;

				layer++;
			}

			if (m_perceptrons.size() == m_nb_perceptrons.size())
			{
#ifdef WINDOWS
				SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), 0x01);
#endif
				cout << "Output not gererate yet" << endl;
#ifdef WINDOWS
				SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), 0x07);
#endif
			}
		}
	}

	void Network::print_output()
	{
		if (m_out.size() > 0)
		{
			for (vector<vector<double> >::iterator it = m_out.begin(); it != m_out.end(); ++it)
			{
				for (vector<double>::iterator it2 = it->begin(); it2 != it->end(); ++it2)
				{
					std::cout << ' ' << *it2;
				}
				std::cout << '\n';
			}
			std::cout << '\n';
		}
		else
		{
#ifdef WINDOWS
			SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), 0x0E);
#endif
			cout << "no output" << endl;
#ifdef WINDOWS
			SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), 0x07);
#endif
		}
	}

	void Network::sim(vector<vector<double>*> P, bool clr)
	{
#ifdef WINDOWS
		SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), 0x0C);
#endif
		if (m_is_train)
		{
			m_out.clear();

			bool same_size = vector_is_uniforme(P);

			if (same_size)
			{

				bool input_set = true;

				vector<vector<Perceptron*> >::iterator it_layer0 = m_perceptrons.begin();
				for (vector<Perceptron*>::iterator it_perceptron = it_layer0->begin(); it_perceptron != it_layer0->end(); ++it_perceptron)
					input_set &= (*it_perceptron)->set_input(P);

				if (input_set)
				{

					for (vector<vector<Perceptron*> >::iterator it_layer = m_perceptrons.begin(); it_layer != m_perceptrons.end(); ++it_layer)
						for (vector<Perceptron*>::iterator it_perceptron = it_layer->begin(); it_perceptron != it_layer->end(); ++it_perceptron)
							(*it_perceptron)->activate();

					for (vector<Perceptron*>::iterator it_perceptron = (m_perceptrons.back()).begin(); it_perceptron != (m_perceptrons.back()).end(); ++it_perceptron)
						m_out.push_back((*it_perceptron)->get_output_cpy());

					if (clr)
						clr_internal_values();
				}
				else
					cout << "Sim => Inputs sizes error." << endl;
			}
			else
				cout << "Sim => Inputs sizes are not the same." << endl;
		}
		else
			cout << "Sim => Network not train." << endl;
#ifdef WINDOWS
		SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), 0x07);
#endif
	}

	void Network::clr_internal_values()
	{
		for (vector<vector<Perceptron*> >::iterator it_layer = m_perceptrons.begin(); it_layer != m_perceptrons.end(); ++it_layer)
			for (vector<Perceptron*>::iterator it_perceptron = it_layer->begin(); it_perceptron != it_layer->end(); ++it_perceptron)
				(*it_perceptron)->clr();
	}

	void Network::generate_network()
	{
		int layer = 0;
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

			double param = 0;
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
	}

	void Network::generate_copy_network(Network const& network)
	{
		int layer = 0;
		for (vector<vector<Perceptron*> >::const_iterator it = network.m_perceptrons.begin(); it != network.m_perceptrons.end(); ++it)
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

			double param = 0;
			if (m_params.size() > 0)
			{
				if (m_params.size() <= 2)
					param = m_params[0];
				else
					param = m_params[layer];
			}

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
#ifdef WINDOWS
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

	bool Network::vector_is_uniforme(vector<vector<double>*>& p_vector)
	{
		bool uniform = true;
		vector<vector<double>*>::iterator it_begin = p_vector.begin();
		int size = (*it_begin)->size();
		for (vector<vector<double>*>::iterator it = it_begin + 1; it != p_vector.end(); ++it)
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

	Perceptron* Network::creat_perceptron(int layer, int id, perceptron_type_t type, double param)
	{
		Perceptron* tmp_perceptron = nullptr;
		switch (type)
		{
		case identities:
			tmp_perceptron = new Perceptron_identity(layer, id);
			break;
		case binary_step:
			tmp_perceptron = new Perceptron_binary_step(layer, id);
			break;
		case logistic:
			tmp_perceptron = new Perceptron_logistic(layer, id);
			break;
		case tanH:
			tmp_perceptron = new Perceptron_tanH(layer, id);
			break;
		case arcTan:
			tmp_perceptron = new Perceptron_arcTan(layer, id);
			break;
		case softsign:
			tmp_perceptron = new Perceptron_softsign(layer, id);
			break;
		case rectifier:
			tmp_perceptron = new Perceptron_rectifier(layer, id);
			break;
		case rectifier_param:
			tmp_perceptron = new Perceptron_rectifier_param(layer, id, param);
			break;
		case ELU:
			tmp_perceptron = new Perceptron_ELU(layer, id, param);
			break;
		case softPlus:
			tmp_perceptron = new Perceptron_softPlus(layer, id);
			break;
		case bent_identity:
			tmp_perceptron = new Perceptron_bent_identity(layer, id);
			break;
		case sinusoid:
			tmp_perceptron = new Perceptron_sinusoid(layer, id);
			break;
		case sinc:
			tmp_perceptron = new Perceptron_sinc(layer, id);
			break;
		case gaussian:
			tmp_perceptron = new Perceptron_gaussian(layer, id);
			break;
		default:
			tmp_perceptron = new Perceptron_identity(layer, id);
			break;
		}
		return tmp_perceptron;
	}

	Perceptron* Network::copy_perceptron(Perceptron& perceptron)
	{
		Perceptron* tmp_perceptron = nullptr;
		if (perceptron.get_type() == "identity")
			tmp_perceptron = new Perceptron_identity((Perceptron_identity&)perceptron);
		else if (perceptron.get_type() == "binary_step")
			tmp_perceptron = new Perceptron_binary_step((Perceptron_binary_step&)perceptron);
		else if (perceptron.get_type() == "logistic")
			tmp_perceptron = new Perceptron_logistic((Perceptron_logistic&)perceptron);
		else if (perceptron.get_type() == "tanH")
			tmp_perceptron = new Perceptron_tanH((Perceptron_tanH&)perceptron);
		else if (perceptron.get_type() == "arcTan")
			tmp_perceptron = new Perceptron_arcTan((Perceptron_arcTan&)perceptron);
		else if (perceptron.get_type() == "softsign")
			tmp_perceptron = new Perceptron_softsign((Perceptron_softsign&)perceptron);
		else if (perceptron.get_type() == "rectifier")
			tmp_perceptron = new Perceptron_rectifier((Perceptron_rectifier&)perceptron);
		else if (perceptron.get_type() == "rectifier_param")
			tmp_perceptron = new Perceptron_rectifier_param((Perceptron_rectifier_param&)perceptron);
		else if (perceptron.get_type() == "ELU")
			tmp_perceptron = new Perceptron_ELU((Perceptron_ELU&)perceptron);
		else if (perceptron.get_type() == "softPlus")
			tmp_perceptron = new Perceptron_softPlus((Perceptron_softPlus&)perceptron);
		else if (perceptron.get_type() == "bent_identity")
			tmp_perceptron = new Perceptron_bent_identity((Perceptron_bent_identity&)perceptron);
		else if (perceptron.get_type() == "sinusoid")
			tmp_perceptron = new Perceptron_sinusoid((Perceptron_sinusoid&)perceptron);
		else if (perceptron.get_type() == "sinc")
			tmp_perceptron = new Perceptron_sinc((Perceptron_sinc&)perceptron);
		else if (perceptron.get_type() == "gaussian")
			tmp_perceptron = new Perceptron_gaussian((Perceptron_gaussian&)perceptron);
		else
			tmp_perceptron = new Perceptron_identity((Perceptron_identity&)perceptron);

		return tmp_perceptron;
	}

	void Network::round_output()
	{
		for (vector<vector<double> >::iterator it_vect = m_out.begin(); it_vect != m_out.end(); ++it_vect)
			for (vector<double>::iterator it = it_vect->begin(); it != it_vect->end(); ++it)
				(*it) = std::round(*it);
	}

} // namespace SNN_network