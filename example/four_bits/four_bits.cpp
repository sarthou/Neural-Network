#include "snn/network/Network.h"
#include "snn/trainer/Trainer.h"
#include "snn/utility/Matrix.h"
#include <iostream>

using namespace SNN;

int test()
{	
	/*Create your network*/
	vector<int> nb = { 1 };
	vector<perceptron_type_t> type = { identities };
	vector<float> param = { };
	Network net(nb, type, param);

	net.print();

	/*Configure your training*/

	trainig_config_t config;

	config.debug_level = 2;
	config.debug_file = "debug.txt";

	config.error_type = mae;
	config.stop_error = 0.00001f;
	config.nb_epochs = 5000;
	config.stop_evolution = false;

	config.training_type = GD_adagrad;
	config.step = 0.1f;
	//config.momentum_factor = 0.05;

	Trainer trainer;
	trainer.set_config(config);

	/*Train your network*/

	vector<float> d = { 0, 0, 0, 0, 0, 1, 1, 0 };
	vector<float> c = { 0, 0, 0, 0, 1, 1, 0, 1 };
	vector<float> b = { 0, 0, 1, 1, 0, 0, 0, 1 };
	vector<float> a = { 0, 1, 0, 1, 0, 0, 0, 1 };
	vector<vector<float> > P = { a, b, c, d };

	vector<float> Ta = { 0, 1, 2, 3, 4, 12, 8, 7 };
	vector<vector<float> > T = { Ta };

	trainer.train(&net, Matrix<float>(P.size(), P.at(0).size(), P), Matrix<float>(T.size(), T.at(0).size(), T));

	/*Use your training network*/

	vector<float> d2 = { 1, 0, 0, 0, 1, 0, 0, 1, 1 };
	vector<float> c2 = { 0, 1, 0, 0, 0, 1, 1, 0, 1 };
	vector<float> b2 = { 0, 0, 1, 0, 1, 0, 1, 0, 1 };
	vector<float> a2 = { 0, 0, 0, 1, 0, 1, 0, 1, 1 };
	vector<vector<float> > P2 = { a2, b2, c2, d2 };

	net.sim(&P2);

	net.round_output();
	net.print_output();
	vector<vector<float> > out = net.get_output_cpy();


	/*Additionals features*/
	net.print();

	Network net2;
	net2 = net;
	net2.print();

#ifdef _WIN32
	system("PAUSE"); // only if you use windows
#endif

	return 0;
}