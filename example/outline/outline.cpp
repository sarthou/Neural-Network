#include "snn/network/Network.h"
#include "snn/trainer/Trainer.h"
#include "bitmap/bmp.h"
#include <iostream>
#include <algorithm>

using namespace SNN;

bool Get_P_T(vector<vector<double> >* T, vector<vector<double> >* P)
{
	int half_size = (3 - 1) / 2;

	bmp bmp_editor;
	bmp_t input = bmp_editor.read_bmp("input.bmp");
	bmp_t output = bmp_editor.read_bmp("output.bmp");

	int x = min(input.width, output.width);
	int y = min(output.height, output.height);
	x = 120;
	y = 10;

	if ((T->size() == 1) && (P->size() == 9))
	{
		for (int xi = 1 + half_size; xi < x - half_size; xi++)
			for (int yi = 1 + half_size; yi < y - half_size; yi++)
			{
				T->at(0).push_back(output.image[xi][yi]);
				for(int local_x = -half_size; local_x <= half_size; local_x++)
					for (int local_y = -half_size; local_y <= half_size; local_y++)
					{
						P->at((local_x + half_size) * 3 + (local_y + half_size)).push_back(input.image[xi + local_x][yi + local_y]);
					}
			}
		return true;
	}
	else
		return false;
}

bool generate_img(Network* net, char* file_name)
{
	int half_size = (3 - 1) / 2;

	bmp bmp_editor;
	bmp_t input = bmp_editor.read_bmp(file_name);

	int x = input.width;
	int y = input.height;

	vector<vector<double> > P(9, vector<double>());

	for (int xi = 1 + half_size; xi < x - half_size; xi++)
		for (int yi = 1 + half_size; y - half_size; yi++)
		{
			for (int local_x = -half_size; local_x < half_size; local_x++)
				for (int local_y = -half_size; local_y < half_size; local_y++)
					P[x*3+y].push_back(input.image[xi + local_x][yi + local_y]);
		}

	net->sim(&P);

	net->round_output();
	vector<vector<double> > out = net->get_output_cpy();

	/*out = answer * 255;
	out = reshape(out, (y - 2 * half_S), (x - 2 * half_S));
	imwrite(out','out.tif')*/

	return true;
}

int main()
{
	/*Create your network*/
	vector<int> nb = { 5, 5 };
	vector<perceptron_type_t> type = { logistic, identities };
	vector<double> param = {};
	Network net(nb, type, param);

	net.print();

	/*Configure your training*/

	trainig_config_t config;

	config.debug_level = 1;
	config.debug_file = "debug.txt";

	config.error_type = mae;
	config.stop_error = 0.01;
	config.nb_epochs = 5000;
	config.stop_evolution = true;

	config.training_type = GD_adagrad;
	config.step = 0.01;
	//config.momentum_factor = 0.05;

	Trainer trainer;
	trainer.set_config(config);

	/*Train your network*/

	vector<vector<double> > P(9, vector<double>());
	vector<vector<double> > T(1, vector<double>());

	Get_P_T(&T, &P);

	std::cout << "train" << std::endl;

	trainer.train(&net, P, T);

	config.training_type = Steepest_descent;
	config.step = 0.01;
	config.stop_evolution = false;

	trainer.set_config(config);
	trainer.train(&net, P, T);

	/*Use your training network*/

	generate_img(&net, "test.bmp");


	/*Additionals features*/
	/*net.print();

	Network net2;
	net2 = net;
	net2.print();*/

	system("PAUSE"); // only if you use windows

	return 0;
}