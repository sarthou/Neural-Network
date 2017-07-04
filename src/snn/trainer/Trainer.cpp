//============================================================================
// Name        : Trainer.cpp
// Authors     : Guillaume Sarthou
// EMail       : open.pode@gmail.com
// Date		   : 29 jun. 2017
// Version     : V1.5
// Copyright   : This file is part of SNN_network project which is released under
//               MIT license.
//============================================================================
#include "snn/trainer/Trainer.h"
#include <cmath>
#include <random>

#ifdef _WIN32
#include <Windows.h>
#endif

#include <chrono>
#include <ctime>

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

	void Trainer::train(Network* net, vector<vector<double> >& P, vector<vector<double> >& T)
	{
		m_P = P;
		m_T = T;
		m_net = net;

		if (can_be_train())
		{
			init_train();
			unsigned int vect_size = m_P.back().size();
			ptr_perceptrons = m_net->m_perceptrons;

			int m_current_layer;
			unsigned int m_current_id;
			unsigned int m_nb_layer = ptr_perceptrons.size() - 1;
			vector<Trainig_process*> empty_process;

			bool small_error = false;
			for (unsigned int nb_epochs = 0; (nb_epochs < m_config.nb_epochs) && !small_error; nb_epochs++)
			{
				time_t start, end;
				start = clock();

				randomise();

				for (unsigned int index = 0; index < vect_size; index++)
				{
					select_single_data(index);
					m_net->sim(&tmp_P, false);

					//set eroor on last layer
					m_current_layer = m_nb_layer - 1; //last layer
					for (m_current_id = 0; m_current_id < ptr_perceptrons[m_current_layer + 1].size(); m_current_id++)
						m_process[m_current_layer][m_current_id]->set_error(tmp_T[m_current_id].front());

					//propagate on all layers
					for (m_current_layer = m_nb_layer - 1; m_current_layer >= 0; m_current_layer--)
					{
						bool is_last = m_current_layer == int(ptr_perceptrons.size() - 2);
						for (m_current_id = 0; m_current_id < ptr_perceptrons[m_current_layer + 1].size(); m_current_id++)
						{
							if (m_current_layer > 0)
								m_process[m_current_layer][m_current_id]->propagate(&m_process[m_current_layer - 1], is_last);
							else
								m_process[m_current_layer][m_current_id]->propagate(&empty_process, is_last);
						}
					}

					//compute on all layers
					for (m_current_layer = 0; (unsigned int)m_current_layer < m_nb_layer; m_current_layer++)
						for (m_current_id = 0; m_current_id < ptr_perceptrons[m_current_layer + 1].size(); m_current_id++)
							m_process[m_current_layer][m_current_id]->compute();

					m_net->clr_internal_values();
				}

				m_net->sim(&m_P);
				compute_error();


				if (m_config.debug_level)
					cout << "epoch : " << nb_epochs + 1 << " => error " << m_error << endl;
				if (m_config.debug_level > 1)
					m_debug_file << m_error << endl;
				if (m_error < m_config.stop_error)
					small_error = true;

				/*detect no evolution*/
				m_mean_error += m_error;
				m_stop_vector.push_back(m_mean_error / (nb_epochs + 1));
				if (m_stop_vector.size() == 9)
				{
					m_stop += m_stop_vector[0];
					m_stop_vector.erase(m_stop_vector.begin());
				}
				m_stop = -m_stop + m_mean_error / (nb_epochs + 1);

				if ((abs(m_stop) < m_config.stop_error) && m_dont_evolve && m_config.stop_evolution)
					small_error = true; //break the training process
				else if (abs(m_stop) < m_config.stop_error)
					m_dont_evolve = true;

				end = clock();
				long long int elapsed_seconds = 1000.f * (double)(end - start) / CLOCKS_PER_SEC;
				
				std::cout << "time: " << elapsed_seconds << endl;
			}
			close_train();
		}
	}

	void Trainer::init_train()
	{
		if (!m_net->m_is_train)
		{
			set_input_perceptrons();
			set_output_perceptrons();
		}

		set_as_pointer();

		set_input();
		set_default_configuration();
		set_trainig_process();

		if (!m_net->m_is_train)
			init_weigh();

		tmp_P.resize(m_P.size());
		tmp_T.resize(m_T.size());
		for (vector<vector<double> >::iterator it = tmp_P.begin(); it != tmp_P.end(); ++it)
			(*it) = vector<double>(1, 0.);
		for (vector<vector<double> >::iterator it = tmp_T.begin(); it != tmp_T.end(); ++it)
			(*it) = vector<double>(1, 0.);

		m_mean_error = 0;
		m_dont_evolve = false;

		m_net->set_it_train();

		if (m_config.debug_level > 1)
			m_debug_file.open(m_config.debug_file);
	}

	void Trainer::close_train()
	{
		for (vector<vector<double>*>::iterator it = m_P_ptr.begin(); it != m_P_ptr.end(); ++it)
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
#ifdef _WIN32
		SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), 0x0C);
#endif
		if (m_net->is_configure())
		{
			if (m_net->m_is_train)
			{
#ifdef WINDOWS
				SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), 0x01);
#endif
				cout << "Network will be re-train" << endl;
#ifdef WINDOWS
				SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), 0x0C);
#endif
			}

			if (m_P.size() != 0)
			{
				if (m_T.size() != 0)
				{
					bool uniform = vector_is_uniforme(m_P);
					if (uniform)
					{
						uniform = vector_is_uniforme(m_T);
						if (uniform)
						{
							if (m_P.back().size() == m_T.back().size())
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
#ifdef _WIN32
		SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), 0x07);
#endif

		return can_be;
	}

	void Trainer::set_input_perceptrons()
	{
		if (m_net->m_perceptrons.size() == m_net->m_nb_perceptrons.size())
		{
			//creat input perceptrons vector
			vector<Perceptron*> input_vect;
			for (unsigned int i = 0; i < m_P.size(); i++)
				input_vect.push_back(new Perceptron_input(-1, i));
			
			m_net->m_perceptrons.insert(m_net->m_perceptrons.begin(), input_vect);

			//link with others layers
			if (m_net->m_nb_perceptrons.size() != 0)
				for (unsigned int i = 0; i < m_net->m_perceptrons[1].size(); i++)
					m_net->m_perceptrons[1][i]->set_input(&(m_net->m_perceptrons[0]));
		}
	}

	void Trainer::set_output_perceptrons()
	{
		if (m_net->m_perceptrons.size() == m_net->m_nb_perceptrons.size() + 1)
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
				if (m_net->m_perceptrons.size() != 0)
				{
					vector<Perceptron*> tmp = m_net->m_perceptrons[m_net->m_nb_perceptrons.size()];
					perceptron->set_input(&(*(m_net->m_perceptrons.end() - 2)));
				}

				m_net->m_perceptrons.back().push_back(perceptron);
			}
		}
	}

	void Trainer::set_input()
	{
		vector<vector<Perceptron*>>::iterator it_input_layer = m_net->m_perceptrons.begin();
		for (vector<Perceptron*>::iterator it_perceptron = it_input_layer->begin(); it_perceptron != it_input_layer->end(); ++it_perceptron)
			(*it_perceptron)->set_input(m_P_ptr);
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
		m_process.resize(m_net->m_perceptrons.size() - 1); //don't create process for input layer
		for (m_current_layer = 0; m_current_layer < m_net->m_perceptrons.size() -1; m_current_layer++)
		{
			m_process[m_current_layer].resize(m_net->m_perceptrons[m_current_layer + 1].size());
			for (m_current_id = 0; m_current_id < m_net->m_perceptrons[m_current_layer + 1].size(); m_current_id++)
				m_process[m_current_layer][m_current_id] = creat_process(m_net->m_perceptrons[m_current_layer + 1][m_current_id]);
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
		unsigned long int vect_size = m_P.back().size();

		default_random_engine generator;
		uniform_int_distribution<unsigned long int> distribution(0, vect_size - 1);
		random_device rd;
		generator.seed(rd());

		unsigned long int from = 0;
		unsigned long int to = 0;
		double tmp_value = 0.;

		for (index = 0; index < vect_size; index++)
		{
			from = distribution(generator);
			to = distribution(generator);

			for (unsigned int i = 0; i < m_P.size(); i++)
			{
				tmp_value = m_P[i][from];
				m_P[i][from] = m_P[i][to];
				m_P[i][to] = tmp_value;
			}

			for (unsigned int i = 0; i < m_T.size(); i++)
			{
				tmp_value = m_T[i][from];
				m_T[i][from] = m_T[i][to];
				m_T[i][to] = tmp_value;
			}
		}
	}

	void Trainer::select_single_data(unsigned int p_index)
	{
		for (unsigned int i = 0; i < m_P.size(); i++)
			tmp_P[i][0] = m_P[i][p_index];

		for (unsigned int i = 0; i < m_T.size(); i++)
			tmp_T[i][0] = m_T[i][p_index];
	}

	void Trainer::compute_error()
	{
		unsigned long int cpt = 0;
		m_error = 0.;

		if (m_config.error_type == mae)
		{
			vector<vector<double> >* output = m_net->get_output();
			for (unsigned int vect_i = 0; vect_i < m_T.size(); vect_i++)
			{
				for (unsigned int i = 0; i < m_T[vect_i].size(); i++)
				{
					m_error += abs((*output)[vect_i][i] - m_T[vect_i][i]);
					cpt++;
				}
			}
			m_error = abs(m_error);
		}
		else
		{
			vector<vector<double> >* output = m_net->get_output();
			for (unsigned int vect_i = 0; vect_i < m_T.size(); vect_i++)
			{
				for (unsigned int i = 0; i < m_T[vect_i].size(); i++)
				{
					m_error += ((*output)[vect_i][i] - m_T[vect_i][i])*((*output)[vect_i][i] - m_T[vect_i][i]);
					cpt++;
				}
			}
		}
		m_error = m_error / cpt;
	}

	void Trainer::set_as_pointer()
	{
		
		for (unsigned int i = 0; i < m_P.size(); i++)
			m_P_ptr.push_back(new vector<double>(m_P[i]));
	}

	bool Trainer::vector_is_uniforme(vector<vector<double> >& p_vector)
	{
		bool uniform = true;
		vector<vector<double> >::iterator it_begin = p_vector.begin();
		unsigned int size = (*it_begin).size();
		for (vector<vector<double> >::iterator it = it_begin + 1; it != p_vector.end(); ++it)
		{
			if (size != (*it).size())
				uniform = false;
		}

		if (p_vector.size() == 0)
			uniform = false;

		return uniform;
	}

} // namespace SNN_network