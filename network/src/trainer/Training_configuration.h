//============================================================================
// Name        : Training_configuration.h
// Authors     : Guillaume Sarthou
// EMail       : open.pode@gmail.com
// Date		   : 11 jun. 2017
// Version     : V1.4
// Copyright   : This file is part of SNN_network project which is released under
//               MIT license.
//============================================================================
#pragma once

#ifndef TRAINING_CONFIGURATION_H
#define TRAINING_CONFIGURATION_H

#include <string>

namespace SNN_network
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
		//general configuration
		unsigned int nb_epochs = 50;
		double step = INFINITY;
		double stop_error = 0.1;
		trainig_type_t training_type = Steepest_descent;
		error_type_t error_type = mse;

		//momentum configuration
		float momentum_factor = INFINITY;

		//debug configuration
		unsigned int debug_level = 0;
		string debug_file = "debug.txt";
	};

} //namespace SNN_network

#endif
