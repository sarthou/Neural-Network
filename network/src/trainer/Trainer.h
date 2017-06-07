//============================================================================
// Name        : Trainer.h
// Authors     : Guillaume Sarthou
// EMail       : open.pode@gmail.com
// Date		   : 6 jun. 2017
// Version     : V1.2
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
#include <vector>

#include <iostream>
#include <fstream>

namespace SNN_network
{

	enum trainig_type_t
	{
		Steepest_descent,
		GD_momentum,
		GD_nesterov
	};

	enum error_type_t
	{
		mae,
		mse
	};

	struct trainig_config_t
	{
		//general configuration
		unsigned int nb_epochs = 50;
		double step = 0.001;
		double stop_error = 0.1;
		trainig_type_t training_type = Steepest_descent;
		error_type_t error_type = mse;

		//momentum configuration
		float momentum_factor = 0.9;

		//debug configuration
		unsigned int debug_level = 0;
		string debug_file = "debug.txt";
	};

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

		ofstream m_debug_file;

		void init_train();
		void close_train();
		bool can_be_train();

		void set_output_perceptrons();
		void set_input();

		void set_trainig_process();
		Trainig_process* creat_process(SNN_network::Perceptron* p_instances);

		void init_weigh();
		void randomise();

		void select_single_data(unsigned int p_index);

		void compute_error();
	};

} // namespace SNN_network

#endif