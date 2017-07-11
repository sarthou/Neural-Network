//============================================================================
// Name        : Trainer.h
// Authors     : Guillaume Sarthou
// EMail       : open.pode@gmail.com
// Date		   : 29 jun. 2017
// Version     : V1.5
// Copyright   : This file is part of SNN_network project which is released under
//               MIT license.
//============================================================================

#ifndef TRAINER_H
#define TRAINER_H

#include "snn/network/Network.h"
#include "snn/trainer/training_process/GD_adagrad_process.h"
#include "snn/trainer/training_process/GD_adam_process.h"
#include "snn/trainer/training_process/GD_rmsprop_process.h"
#include "snn/trainer/training_process/GD_nesterov_process.h"
#include "snn/trainer/training_process/GD_momentum_process.h"
#include "snn/trainer/training_process/Steepest_descent_process.h"

#include "snn/utility/Matrix.h"
#include <vector>

#include <iostream>
#include <fstream>

#include "snn/trainer/Training_configuration.h"

namespace SNN
{

	class Trainer
	{
	public:
		Trainer();
		~Trainer();

		void train(Network* net, Matrix<float>& P, Matrix<float>& T);

		void set_config(trainig_config_t p_config) { m_config = p_config; };

	private:
		Network* m_net;
		vector<vector<Perceptron*>> ptr_perceptrons;
		vector<vector<Trainig_process*>> m_process;

		trainig_config_t m_config;
		float m_error;

		vector<float> m_stop_vector;
		float m_stop;
		float m_mean_error;
		bool m_dont_evolve;

		ofstream m_debug_file;

		void init_train(Matrix<float>& P, Matrix<float>& T);
		void close_train();
		bool can_be_train(Matrix<float>& P, Matrix<float>& T);

		void set_input_perceptrons(Matrix<float>& P);
		void set_output_perceptrons(Matrix<float>& T);

		void set_default_configuration();
		void set_trainig_process();
		Trainig_process* creat_process(SNN::Perceptron* p_instances);

		void init_weigh();
		void randomise(Matrix<float>& P, Matrix<float>& T);

		void select_single_data(unsigned int p_index, Matrix<float>& singleP, Matrix<float>& singleT, Matrix<float>& P, Matrix<float>& T);

		void compute_error(Matrix<float>& T);
	};

} // namespace SNN_network

#endif