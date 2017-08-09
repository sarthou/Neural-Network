#include "snn/network/Network.h"
#include "snn/trainer/Trainer.h"
#include "snn/utility/Matrix.h"
#include <iostream>

#include "snn/serializer/Src_serializer.h"
#include "snn/serializer/Bin_serializer.h"

using namespace SNN;

void display_pannel(Matrix<float>& pannel)
{
	for (unsigned int col = 0; col < 7; col++)
	{
		cout << "|" << col << "\t";
	}
	cout << endl;
	for (int row = 5; row >= 0; row--)
	{
		for (unsigned int col = 0; col < 7; col++)
		{
			cout << "|";
			if (pannel(row, col) == -1)
				cout << "X\t";
			else if (pannel(row, col) == 1)
				cout << "O\t";
			else
				cout << " \t";
		}
		cout << endl << "-----------------------------------------------------" << endl;
	}
}

int play(int player, Matrix<float>& pannel, vector<int>& top)
{
	int play_col;
	bool ok;
	do
	{
		ok = true;
		cin >> play_col;
		if (play_col < 0 || play_col > 6)
			ok = false;
		else if (top[play_col] >= 6)
			ok = false;
	} while (!ok);


	pannel(top[play_col], play_col) = player;
	top[play_col]++;
	display_pannel(pannel);

	return play_col;
}

void play(int player, Matrix<float>& pannel, vector<int>& top, int choise)
{
	int play_col = choise;

	pannel(top[play_col], play_col) = player;
	top[play_col]++;
	display_pannel(pannel);
}

void setP(Matrix<float>& P, Matrix<float>& pannel, vector<int> top, int player)
{
	for (unsigned int i = 0; i < 4; i++)
	{
		int min = 10;
		int min_index = -1;
		for(unsigned int j = 0; j < 4; j++)
			if (top[j + i] < min)
			{
				min = top[j + i];
				min_index = j + i - 4;
				if (min_index < 0)
					min_index = 0;
			}

		for(unsigned int row = 0; row < 3; row++)
			for (unsigned int col = 0; col < 4; col++)
			{
				P(row * 4 + col, i) = pannel(row + min_index, col + i) * player;
			}
	}
}

void setP2(Matrix<float>& P2, vector<vector<float>> out)
{
	for (unsigned int i = 0; i < 4; i++)
		for (unsigned int j = 0; j < 4; j++)
			P2(i * 4 + j, 0) = out[i][j];
}

void setT(Matrix<float>& T, int choise)
{
	for (unsigned int i = 0; i < 4; i++)
		for (unsigned int col = 0; col < 4; col++)
		{
			if (i + col == choise)
				T(col, i) = 1;
			else
				T(col, i) = 0;
		}
}

void setT2(Matrix<float>& T, int choise)
{
	for (unsigned int i = 0; i < 7; i++)
		if (i == choise)
			T(i, 0) = 1;
		else
			T(i, 0) = 0;
}

void print_output(vector<vector<float>> out)
{
	for (unsigned int i = 0; i < 4; i++)
	{
		for (unsigned int j = 0; j < i; j++)
			cout << "  \t";

		for (unsigned int j = 0; j < 4; j++)
			cout << " " << out[i][j] << "\t";

		cout << endl;
	}
}

int get_result(vector<vector<float>> data)
{
	vector<float> result;
	result.resize(7);

	result[0] = data[0][0];
	result[1] = fmaxf(data[0][1], data[1][0]);
	result[2] = fmaxf(data[0][2], fmaxf(data[1][1], data[2][0]));
	result[3] = fmaxf(data[0][3], fmaxf(data[1][2], fmaxf(data[2][1], data[3][0])));
	result[4] = fmaxf(data[1][3], fmaxf(data[2][2], data[3][1]));
	result[5] = fmaxf(data[2][3], data[3][2]);
	result[6] = data[3][3];

	for (unsigned int i = 0; i < 7; i++)
		cout << result[i] << " : ";
	cout << endl;

	float max = 0;
	int index = -1;
	for (unsigned int i = 0; i < 7; i++)
		if (result[i] >= max)
		{
			max = result[i];
			index = i;
		}

	return index;
}

int main()
{
	/*Create your network*/
	vector<int> nb = { 10, 7 };
	vector<perceptron_type_t> type = { logistic, identities, gaussian };
	vector<float> param = {};
	Network net(nb, type, param);

	/*Configure your training*/

	trainig_config_t config;

	config.debug_level = 0;

	config.error_type = mae;
	config.stop_error = 0.00001f;
	config.nb_epochs = 1;
	config.stop_evolution = false;

	config.training_type = Steepest_descent;
	config.step = 0.001f;
	//config.momentum_factor = 0.05;

	Trainer trainer;
	trainer.set_config(config);

	/*Train your network*/

	Matrix<float> pannel(6, 7);
	for (unsigned int i = 0; i < 6 * 7; i++)
		pannel[i] = 0;

	vector<int> top(7, 0);

	Matrix<float> P(4*4, 4);
	Matrix<float> T(4, 4);

	Matrix<float> P2(4 * 4, 1);
	Matrix<float> T2(7, 1);

	Bin_serializer serial;
	serial.load("four.bin", net);
	//serial.load("four2.bin", net2);

	display_pannel(pannel);
	int player = 1;
	for (unsigned int i = 0; i < 6 * 7; i++)
	{
		setP(P, pannel, top, player);
		int choise = play(player, pannel, top);
		setT(T, choise);
		trainer.set_config(config);
		trainer.train(&net, P, T);

		net.sim(P);
		vector<vector<float>> out = net.get_output_cpy();
		
		player = -player;

		setP(P, pannel, top, player);
		net.sim(P);
		out = net.get_output_cpy();

		int ia_choise = get_result(out);
		play(player, pannel, top, ia_choise);

		player = -player;

		serial.save("four.bin", net);
		//serial.save("four2.bin", net2);
	}


#ifdef _WIN32
	system("PAUSE"); // only if you use windows
#endif

	return 0;
}