#include "perceptron\Perceptron.h"
#include "perceptron\Perceptrons.h"
#include "network\Network.h"
#include "trainer\Trainer.h"
#include <iostream>

using namespace SNN_network;

int main()
{
	Trainer trainer;
		
	vector<int> nb = { 10, 10, 10};//, 10
	vector<perceptron_type_t> type = { logistic, identities };//identities logistic
	vector<double> param = { 1.};
	Network net(nb, type);
	net.print();

	vector<double> a = { 0, 0, 0, 0, 1, 1, 1, 1 };
	vector<double> b = { 0, 0, 1, 1, 0, 0, 0, 1 };
	vector<double> c = { 0, 1, 0, 1, 0, 0, 0, 1 };
	vector<vector<double>*> P1 = { &a, &b, &c};

	vector<double> Ta = { 0, 1, 2, 3, 4, 4, 4, 7 };
	vector<double> Tb = { 0, 0, 0, 0, 0, 0, 2, 2 };
	vector<double> Tc = { 0, 3, 3, 0, 3, 0, 0, 3 };
	vector<double> Td = { 0, 1, 1, 2, 1, 2, 2, 3 };
	vector<vector<double>*> T1 = {&Ta/*, &Tb, &Tc, &Td*/};

	vector<double> a2 = { 0, 0, 0, 1, 1, 0, 0, 0, 1 };
	vector<double> b2 = { 0, 0, 0, 1, 1, 0, 0, 0, 1 };
	vector<double> c2 = { 0, 0, 1, 1, 0, 0, 0, 0, 1 };
	vector<vector<double>*> P2 = { &a2, &b2, &c2 };

	vector<double> a3 = { 1, 0, 0, 0, 1, 1, 1, 1 };
	vector<double> b3 = { 0, 0, 1, 1, 1, 0, 1, 1 };
	vector<double> c3 = { 0, 1, 0, 1, 0, 1, 0, 1 };
	vector<vector<double>*> P3 = { &a3, &b3, &c3 };

	trainig_config_t config;
	config.error_type = mae;
	config.nb_epochs = 100000;
	config.step = 0.1;
	config.stop_error = 0.00000001;
	config.debug_level = 1;

	trainer.set_config(config);
	trainer.train(&net, P1, T1);
	net.print();

	net.sim(P2);
	net.print_output();
	net.round_output();
	net.print_output();

	net.sim(P3);
	net.print_output();
	net.round_output();
	net.print_output();

	system("PAUSE");

	return 0;
}