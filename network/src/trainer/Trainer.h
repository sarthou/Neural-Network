//============================================================================
// Name        : Trainer.h
// Authors     : Guillaume Sarthou
// EMail       : open.pode@gmail.com
// Date		   : 25 jun. 2017
// Version     : V1.4
// Copyright   : This file is part of SNN_network project which is released under
//               MIT license.
//============================================================================
#pragma once

#ifndef TRAINER_H
#define TRAINER_H

#include "../network/Network.h"
#include "training process\Steepest_descent_process.h"
#include "training process\GD_momentum_process.h"
#include "training process\GD_nesterov_process.h"
#include "training process\GD_adagrad_process.h"
#include "training process\GD_rmsprop_process.h"
#include "training process\GD_adam_process.h"
#include <vector>

#include <iostream>
#include <fstream>

#include "Training_configuration.h"

namespace SNN
{

	class Trainer
	{
	public:
		Trainer();
		~Trainer();

		void train(Network* net, vector<vector<double>*>& P, vector<vector<double>*>& T);

		void set_config(trainig_config_t p_config) { m_config = p_config; };

	private:
		Network* m_net;
		vector<vector<Perceptron*>> ptr_perceptrons;
		vector<vector<Trainig_process*>> m_process;

		vector<vector<double>*> m_P;
		vector<vector<double>*> m_T;
		vector<vector<double>*> tmp_P;
		vector<vector<double>*> tmp_T;

		trainig_config_t m_config;
		double m_error;

		vector<double> m_stop_vector;
		double m_stop;
		double m_mean_error;
		bool m_dont_evolve;

		ofstream m_debug_file;

		void init_train();
		void close_train();
		bool can_be_train();

		void set_output_perceptrons();
		void set_input();

		void set_default_configuration();
		void set_trainig_process();
		Trainig_process* creat_process(SNN::Perceptron* p_instances);

		void init_weigh();
		void randomise();

		void select_single_data(unsigned int p_index);

		void compute_error();
	};

} // namespace SNN_network

#endif
