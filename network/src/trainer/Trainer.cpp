//============================================================================
// Name        : Trainer.cpp
// Authors     : Guillaume Sarthou
// EMail       : open.pode@gmail.com
// Date		   : 25 jun. 2017
// Version     : V1.4
// Copyright   : This file is part of SNN_network project which is released under
//               MIT license.
//============================================================================
#include "Trainer.h"
#include <windows.h>
#include <random>

namespace SNN
{

	Trainer::Trainer()
	{
		m_net = nullptr;

		m_config.nb_epochs = 50;
		m_config.step = 0.1;
		m_config.stop_error = 0.1;
		m_config.training_type = Steepest_descent;
		m_config.error_type = mae;

		m_error = 0.;
		m_stop = 0.;
	}

	Trainer::~Trainer()
	{

	}

	void Trainer::train(Network* net, vector<vector<double>*>& P, vector<vector<double>*>& T)
	{
		m_P = P;
		m_T = T;
		m_net = net;

		if (can_be_train())
		{
			init_train();
			unsigned int vect_size = m_P.back()->size();
			ptr_perceptrons = m_net->m_perceptrons;

			int m_current_layer;
			unsigned int m_current_id;
			unsigned int m_nb_layer = ptr_perceptrons.size();

			bool small_error = false;
			for (unsigned int nb_epochs = 0; (nb_epochs < m_config.nb_epochs) && !small_error; nb_epochs++)
			{
				randomise();

				for (unsigned int index = 0; index < vect_size; index++)
				{
					select_single_data(index);
					m_net->sim(tmp_P, false);

					m_current_layer = m_nb_layer - 1;
					for (m_current_id = 0; m_current_id < ptr_perceptrons[m_current_layer].size(); m_current_id++)
						m_process[m_current_layer][m_current_id]->set_error(tmp_T[m_current_id]->front());

					for (m_current_layer = m_nb_layer - 1; m_current_layer >= 0; m_current_layer--)
						for (m_current_id = 0; m_current_id < ptr_perceptrons[m_current_layer].size(); m_current_id++)
						{
							if (m_current_layer > 0)
								m_process[m_current_layer][m_current_id]->propagate(m_process[m_current_layer - 1], m_current_layer == ptr_perceptrons.size() - 1);
							else
							{
								vector<Trainig_process*> tmp;
								m_process[m_current_layer][m_current_id]->propagate(tmp, m_current_layer == ptr_perceptrons.size() - 1);
							}
						}

					for (m_current_layer = 0; (unsigned int)m_current_layer < m_nb_layer; m_current_layer++)
						for (m_current_id = 0; m_current_id < ptr_perceptrons[m_current_layer].size(); m_current_id++)
							m_process[m_current_layer][m_current_id]->compute();

					//std::cout << "-----" << std::endl;

					m_net->clr_internal_values();
				}

				m_net->sim(m_P);
				compute_error();

				/////
				m_stop_vector.push_back(m_error);
				if (m_stop_vector.size() == 3)
				{
					m_stop += m_stop_vector[0];
					m_stop_vector.erase(m_stop_vector.begin());
				}
				m_stop = -m_stop + m_error;
				/////

				if (m_config.debug_level)
					cout << "epoch : " << nb_epochs + 1 << " => error " << m_error << endl;
				if (m_config.debug_level > 1)
					m_debug_file << fabs(m_stop)/m_error << endl;
				if (m_error < m_config.stop_error)
					small_error = true;
			}
			close_train();
		}
	}

	void Trainer::init_train()
	{
		if(!m_net->m_is_train)
			set_output_perceptrons();

		set_input();
		set_default_configuration();
		set_trainig_process();

		if (!m_net->m_is_train)
			init_weigh();

		tmp_P.resize(m_P.size());
		tmp_T.resize(m_T.size());
		for (vector<vector<double>*>::iterator it = tmp_P.begin(); it != tmp_P.end(); ++it)
			(*it) = new vector<double>(1, 0.);
		for (vector<vector<double>*>::iterator it = tmp_T.begin(); it != tmp_T.end(); ++it)
			(*it) = new vector<double>(1, 0.);

		m_net->set_it_train();

		if (m_config.debug_level > 1)
			m_debug_file.open(m_config.debug_file);
	}

	void Trainer::close_train()
	{
		for (vector<vector<double>*>::iterator it = tmp_P.begin(); it != tmp_P.end(); ++it)
			delete (*it);
		for (vector<vector<double>*>::iterator it = tmp_T.begin(); it != tmp_T.end(); ++it)
			delete (*it);

		for (vector<vector<Trainig_process*>>::iterator it_layer = m_process.begin(); it_layer != m_process.end(); ++it_layer)
			for (vector<Trainig_process*>::iterator it_process = it_layer->begin(); it_process != it_layer->end(); ++it_process)
				delete (*it_process);

		if (m_config.debug_level > 1)
			m_debug_file.close();

		trainig_config_t tmp_config;
		m_config = tmp_config;
	}

	bool Trainer::can_be_train()
	{
		bool can_be = false;
		SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), 0x0C);
		if (m_net->is_configure())
		{
			if (m_net->m_is_train)
			{
				SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), 0x01);
				cout << "Network will be re-train" << endl;
				SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), 0x0C);
			}

			if (m_P.size() != 0)
			{
				if (m_T.size() != 0)
				{
					bool uniform = m_net->vector_is_uniforme(m_P);
					if (uniform)
					{
						uniform = m_net->vector_is_uniforme(m_T);
						if (uniform)
						{
							if (m_P.back()->size() == m_T.back()->size())
								can_be = true;
							else
								cout << "Trainer => Vectors P and T haven't the same lenght" << endl;
						}
						else
							cout << "Trainer => Targets sizes are not the same." << endl;
					}
					else
						cout << "Trainer => Inputs sizes are not the same." << endl;
				}
				else
					cout << "Trainer => No target detected." << endl;
			}
			else
				cout << "Trainer => No input detected." << endl;
		}
		else
			cout << "Trainer => network not configure" << endl;
		SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), 0x07);

		return can_be;
	}

	void Trainer::set_output_perceptrons()
	{
		if (m_net->m_perceptrons.size() == m_net->m_nb_perceptrons.size())
		{
			perceptron_type_t type;
			if (m_net->m_nb_perceptrons.size() == 0)
			{
				if (m_net->m_types.size() == 0)
					type = binary_step;
				else
					type = m_net->m_types.back();
			}
			else
			{
				if (m_net->m_types.size() == 0)
					type = identities;
				else
					type = m_net->m_types.back();
			}

			double param = 0.;
			if (m_net->m_params.size() > 0)
				param = m_net->m_params.back();

			vector<Perceptron*> temp_vect;
			m_net->m_perceptrons.push_back(temp_vect);
			for (unsigned int i = 0; i < m_T.size(); i++)
			{
				Perceptron* perceptron = m_net->creat_perceptron(m_net->m_nb_perceptrons.size(), i, type, param);
				if (m_net->m_nb_perceptrons.size() != 0)
					perceptron->set_input(&(*(m_net->m_perceptrons.end() - 2)));

				m_net->m_perceptrons.back().push_back(perceptron);
			}
		}
	}

	void Trainer::set_input()
	{
		vector<vector<Perceptron*>>::iterator it_input_layer = m_net->m_perceptrons.begin();
		for (vector<Perceptron*>::iterator it_perceptron = it_input_layer->begin(); it_perceptron != it_input_layer->end(); ++it_perceptron)
			(*it_perceptron)->set_input(m_P);
	}

	void Trainer::set_default_configuration()
	{
		switch (m_config.training_type)
		{
		case Steepest_descent:
			Steepest_descent_process::set_default_configuration(&m_config);
			break;
		case GD_momentum:
			GD_momentum_process::set_default_configuration(&m_config);
			break;
		case GD_nesterov:
			GD_nesterov_process::set_default_configuration(&m_config);
			break;
		case GD_adagrad:
			GD_adagrad_process::set_default_configuration(&m_config);
			break;
		case GD_RMSprop:
			GD_rmsprop_process::set_default_configuration(&m_config);
			break;
		case GD_adam:
			GD_adam_process::set_default_configuration(&m_config);
			break;
		default:
			Steepest_descent_process::set_default_configuration(&m_config);
			break;
		}
	}

	void Trainer::set_trainig_process()
	{
		unsigned int m_current_id, m_current_layer;
		m_process.resize(m_net->m_perceptrons.size());
		for (m_current_layer = 0; m_current_layer < m_net->m_perceptrons.size(); m_current_layer++)
		{
			m_process[m_current_layer].resize(m_net->m_perceptrons[m_current_layer].size());
			for (m_current_id = 0; m_current_id < m_net->m_perceptrons[m_current_layer].size(); m_current_id++)
				m_process[m_current_layer][m_current_id] = creat_process(m_net->m_perceptrons[m_current_layer][m_current_id]);
		}
	}

	Trainig_process* Trainer::creat_process(Perceptron* p_instance)
	{
		Trainig_process* tmp_process;
		switch (m_config.training_type)
		{
		case Steepest_descent:
			tmp_process = new Steepest_descent_process(p_instance, m_config.step);
			break;
		case GD_momentum:
			tmp_process = new GD_momentum_process(p_instance, m_config.step, m_config.momentum_factor);
			break;
		case GD_nesterov:
			tmp_process = new GD_nesterov_process(p_instance, m_config.step, m_config.momentum_factor);
			break;
		case GD_adagrad:
			tmp_process = new GD_adagrad_process(p_instance, m_config.step);
			break;
		case GD_RMSprop:
			tmp_process = new GD_rmsprop_process(p_instance, m_config.step);
			break;
		case GD_adam:
			tmp_process = new GD_adam_process(p_instance, m_config.step);
			break;
		default:
			tmp_process = new Steepest_descent_process(p_instance, m_config.step);
			break;
		}

		return tmp_process;
	}

	void Trainer::init_weigh()
	{
		default_random_engine generator;
		uniform_real_distribution<double> distribution(-0.5, 0.5);
		random_device rd;
		generator.seed(rd());

		vector<double> tmp_weigh;

		for (vector<vector<Perceptron*>>::iterator it_layer = m_net->m_perceptrons.begin(); it_layer != m_net->m_perceptrons.end(); ++it_layer)
			for (vector<Perceptron*>::iterator it_perceptron = it_layer->begin(); it_perceptron != it_layer->end(); ++it_perceptron)
			{
				(*it_perceptron)->set_bia(distribution(generator));
				tmp_weigh = (*it_perceptron)->get_weigh();
				for (vector<double>::iterator w_it = tmp_weigh.begin(); w_it != tmp_weigh.end(); ++w_it)
					(*w_it) = distribution(generator);
				(*it_perceptron)->set_weigh(tmp_weigh);
			}
	}

	void Trainer::randomise()
	{
		unsigned long int index = 0;
		unsigned long int vect_size = m_P.back()->size();

		default_random_engine generator;
		uniform_int_distribution<unsigned long int> distribution(0, vect_size - 1);
		random_device rd;
		generator.seed(rd());

		unsigned long int from = 0;
		unsigned long int to = 0;
		double tmp_value = 0.;
		vector<vector<double>*>::iterator it;

		for (index = 0; index < vect_size; index++)
		{
			from = distribution(generator);
			to = distribution(generator);

			for (it = m_P.begin(); it != m_P.end(); ++it)
			{
				tmp_value = (*(*it))[from];
				(*(*it))[from] = (*(*it))[to];
				(*(*it))[to] = tmp_value;
			}

			for (it = m_T.begin(); it != m_T.end(); ++it)
			{
				tmp_value = (*(*it))[from];
				(*(*it))[from] = (*(*it))[to];
				(*(*it))[to] = tmp_value;
			}
		}
	}

	void Trainer::select_single_data(unsigned int p_index)
	{
		vector<vector<double>*>::iterator it_single = tmp_P.begin();
		for (vector<vector<double>*>::iterator it_ref = m_P.begin(); it_ref != m_P.end(); ++it_ref)
		{
			(*(*it_single))[0] = (*(*it_ref))[p_index];
			++it_single;
		}

		it_single = tmp_T.begin();
		for (vector<vector<double>*>::iterator it_ref = m_T.begin(); it_ref != m_T.end(); ++it_ref)
		{
			(*(*it_single))[0] = (*(*it_ref))[p_index];
			++it_single;
		}
	}

	void Trainer::compute_error()
	{
		unsigned long int cpt = 0;
		m_error = 0.;

		if (m_config.error_type == mae)
		{
			vector<vector<double>>::iterator out_vect_it = m_net->get_output()->begin();
			for (vector<vector<double>*>::iterator in_vect_it = m_T.begin(); in_vect_it != m_T.end(); ++in_vect_it)
			{
				vector<double>::iterator out_it = out_vect_it->begin();
				for (vector<double>::iterator in_it = (*in_vect_it)->begin(); in_it != (*in_vect_it)->end(); ++in_it)
				{
					m_error += abs((*out_it) - (*in_it));
					cpt++;
					++out_it;
				}
				++out_vect_it;
			}
			m_error = abs(m_error);
		}
		else
		{
			vector<vector<double>>::iterator out_vect_it = m_net->get_output()->begin();
			for (vector<vector<double>*>::iterator in_vect_it = m_T.begin(); in_vect_it != m_T.end(); ++in_vect_it)
			{
				vector<double>::iterator out_it = out_vect_it->begin();
				for (vector<double>::iterator in_it = (*in_vect_it)->begin(); in_it != (*in_vect_it)->end(); ++in_it)
				{
					m_error += ((*out_it) - (*in_it))*((*out_it) - (*in_it));
					cpt++;
					++out_it;
				}
				++out_vect_it;
			}
		}
		m_error = m_error / cpt;
	}

} // namespace SNN_network