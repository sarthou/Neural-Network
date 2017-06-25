//============================================================================
// Name        : Training_configuration.h
// Authors     : Guillaume Sarthou
// EMail       : open.pode@gmail.com
// Date		   : 25 jun. 2017
// Version     : V1.4
// Copyright   : This file is part of SNN_network project which is released under
//               MIT license.
//============================================================================
#pragma once

#ifndef TRAINING_CONFIGURATION_H
#define TRAINING_CONFIGURATION_H

#include <string>

namespace SNN
{

	enum trainig_type_t
	{
		Steepest_descent,
		GD_momentum,
		GD_nesterov,
		GD_adagrad,
		GD_RMSprop,
		GD_adam
	};

	enum error_type_t
	{
		mae,
		mse
	};

	struct trainig_config_t
	{
		//debug configuration
		unsigned int debug_level = 0;
		string debug_file = "debug.txt";

		error_type_t error_type = mse;
		double stop_error = 0.00001;
		unsigned int nb_epochs = 5000;
		bool stop_evolution = false;

		trainig_type_t training_type = Steepest_descent;
		double step = INFINITY;
		double momentum_factor = INFINITY;
	};

} //namespace SNN_network

#endif
