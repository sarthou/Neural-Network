//============================================================================
// Name        : Trainer.cpp
// Authors     : Guillaume Sarthou
// EMail       : open.pode@gmail.com
// Date		   : 18 jul. 2017
// Version     : V2.0
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
		m_config.step = 0.1f;
		m_config.stop_error = 0.1f;
		m_config.training_type = Steepest_descent;
		m_config.error_type = mae;

		m_error = 0.f;
		m_stop = 0.f;
	}

	Trainer::~Trainer()
	{

	}

	void Trainer::train(Network* net, Matrix<float>& P, Matrix<float>& T)
	{
		m_net = net;

		if (can_be_train(P, T))
		{
			//create local training data
			Matrix<float> single_P(P.get_row_count(), 1);
			Matrix<float> single_T(T.get_row_count(), 1);

			//init all training
			init_train(P, T);
			
			//create local variables
			unsigned int vect_size = P.get_col_count();
			int m_current_layer;
			unsigned int m_current_id;
			ptr_perceptrons = m_net->m_perceptrons;
			unsigned int m_nb_layer = ptr_perceptrons.size() - 1;
			vector<Trainig_process*> empty_process;	

			bool small_error = false;
			for (unsigned int nb_epochs = 0; (nb_epochs < m_config.nb_epochs) && !small_error; nb_epochs++)
			{
				time_t start, end;
				start = clock();

				randomise(P, T);

				for (unsigned int index = 0; index < vect_size; index++)
				{
					select_single_data(index, single_P, single_T, P, T);
					m_net->sim(single_P, false);

					//set eroor on last layer
					m_current_layer = m_nb_layer - 1; //last layer
					for (m_current_id = 0; m_current_id < ptr_perceptrons[m_current_layer + 1].size(); m_current_id++)
						m_process[m_current_layer][m_current_id]->set_error( *single_T.get_row(m_current_id) );

					//propagate on all layers
					int last_layer_num = int(ptr_perceptrons.size() - 2);
					for (m_current_layer = m_nb_layer - 1; m_current_layer >= 0; m_current_layer--)
					{
						bool is_last = m_current_layer == last_layer_num;
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

				m_net->sim(P);
				compute_error(T);

				//print debug
				if (m_config.debug_level)
					cout << "epoch : " << nb_epochs + 1 << " => error " << m_error << endl;
				if (m_config.debug_level > 1)
					m_debug_file << m_error << endl;

				//test stop condition with error
				if (m_error < m_config.stop_error)
					small_error = true;

				//detect no evolution
				if (m_config.stop_evolution)
				{
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
				}

				end = clock();
				long long int elapsed_seconds = (long long int)(1000.f * (float)(end - start) / CLOCKS_PER_SEC);
				
				std::cout << "time: " << elapsed_seconds << endl;
			}
			close_train();
		}
	}

	void Trainer::init_train(Matrix<float>& P, Matrix<float>& T)
	{
		//finish network creation
		if (!m_net->m_is_train)
		{
			set_input_perceptrons(P);
			set_output_perceptrons(T);
		}

		//reconfigure network for training
		set_default_configuration();
		set_trainig_process();

		//init weigh if new network
		if (!m_net->m_is_train)
			init_weigh();

		//init stop conditions
		m_mean_error = 0;
		m_dont_evolve = false;

		//init data informations
		P_nb_row = P.get_row_count();
		T_nb_row = T.get_row_count();

		//create debug file
		if (m_config.debug_level > 1)
			m_debug_file.open(m_config.debug_file);

		m_net->set_it_train();
	}

	void Trainer::close_train()
	{
		for (vector<vector<Trainig_process*>>::iterator it_layer = m_process.begin(); it_layer != m_process.end(); ++it_layer)
			for (vector<Trainig_process*>::iterator it_process = it_layer->begin(); it_process != it_layer->end(); ++it_process)
				delete (*it_process);

		if (m_config.debug_level > 1)
			m_debug_file.close();

		trainig_config_t tmp_config;
		m_config = tmp_config;
	}

	bool Trainer::can_be_train(Matrix<float>& P, Matrix<float>& T)
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

			if (P.get_row_count() != 0)
			{
				if (T.get_row_count() != 0)
				{
					if (P.get_col_count() == T.get_col_count())
						can_be = true;
					else
						cout << "Trainer => Vectors P and T haven't the same lenght" << endl;
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

	void Trainer::set_input_perceptrons(Matrix<float>& P)
	{
		if (m_net->m_perceptrons[0].size() == 0)
		{
			//creat input perceptrons vector
			for (unsigned int i = 0; i < P.get_row_count(); i++)
				m_net->m_perceptrons[0].push_back(new Perceptron_input(-1, i));
			

			//link with others layers
			if (m_net->m_nb_perceptrons.size() != 0)
				for (unsigned int i = 0; i < m_net->m_perceptrons[1].size(); i++)
					m_net->m_perceptrons[1][i]->set_input(&(m_net->m_perceptrons[0]));
		}
	}

	void Trainer::set_output_perceptrons(Matrix<float>& T)
	{
		if (m_net->m_perceptrons[m_net->m_nb_perceptrons.size() + 1].size() == 0)
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

			float param = 0.;
			if (m_net->m_params.size() > 0)
				param = m_net->m_params.back();

			for (unsigned int i = 0; i < T.get_row_count(); i++)
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
		uniform_real_distribution<float> distribution(-0.5, 0.5);
		random_device rd;
		generator.seed(rd());

		vector<float> tmp_weigh;

		for (vector<vector<Perceptron*>>::iterator it_layer = m_net->m_perceptrons.begin(); it_layer != m_net->m_perceptrons.end(); ++it_layer)
			for (vector<Perceptron*>::iterator it_perceptron = it_layer->begin(); it_perceptron != it_layer->end(); ++it_perceptron)
			{
				(*it_perceptron)->set_bia(distribution(generator));
				tmp_weigh = (*it_perceptron)->get_weigh();
				for (vector<float>::iterator w_it = tmp_weigh.begin(); w_it != tmp_weigh.end(); ++w_it)
					(*w_it) = distribution(generator);
				(*it_perceptron)->set_weigh(tmp_weigh);
			}
	}

	void Trainer::randomise(Matrix<float>& P, Matrix<float>& T)
	{
		unsigned long int index = 0;
		unsigned long int vect_size = P.get_col_count();

		default_random_engine generator;
		uniform_int_distribution<unsigned long int> distribution(0, vect_size - 1);
		random_device rd;
		generator.seed(rd());

		unsigned long int from = 0;
		unsigned long int to = 0;
		float tmp_value = 0.;
		unsigned int i = 0;

		for (index = 0; index < vect_size / 2; index++)
		{
			from = distribution(generator);
			to = distribution(generator);

			for (i = 0; i < P_nb_row; i++)
			{
				tmp_value = P(i,from);
				P(i, from) = P(i,to);
				P(i, to) = tmp_value;
			}

			for (i = 0; i < T_nb_row; i++)
			{
				tmp_value = T(i, from);
				T(i, from) = T(i, to);
				T(i, to) = tmp_value;
			}
		}
	}

	void Trainer::select_single_data(unsigned int p_index, Matrix<float>& singleP, Matrix<float>& singleT, Matrix<float>& P, Matrix<float>& T)
	{
		for (unsigned int i = 0; i < P_nb_row; i++)
			singleP[i] = P(i,p_index);

		for (unsigned int i = 0; i < T_nb_row; i++)
			singleT[i] = T(i,p_index);
	}

	void Trainer::compute_error(Matrix<float>& T)
	{
		m_error = 0.;
		unsigned int col_count = T.get_col_count();
		unsigned long int cpt = T_nb_row*col_count;

		if (m_config.error_type == mae)
		{
			vector<vector<float> >* output = m_net->get_output();
			for (unsigned int vect_i = 0; vect_i < T_nb_row; vect_i++)
			{
				for (unsigned int i = 0; i < col_count; i++)
					m_error += abs((*output)[vect_i][i] - T(vect_i,i));
			}
		}
		else
		{
			vector<vector<float> >* output = m_net->get_output();
			for (unsigned int vect_i = 0; vect_i < T_nb_row; vect_i++)
			{
				for (unsigned int i = 0; i < col_count; i++)
					m_error += ((*output)[vect_i][i] - T(vect_i,i))*((*output)[vect_i][i] - T(vect_i,i));
			}
		}
		m_error = m_error / cpt;
	}

} // namespace SNN_network