//============================================================================
// Name        : Perceptrons.cpp
// Authors     : Guillaume Sarthou
// EMail       : open.pode@gmail.com
// Date		   : 29 jun. 2017
// Version     : V1.5
// Copyright   : This file is part of SNN_network project which is released under
//               MIT license.
//============================================================================
#include "snn/perceptron/Perceptron.h"
#include "snn/perceptron/Perceptrons.h"

#include <cmath>

namespace SNN
{
	/*Perceptron identity*/
	void activate_identities(vector<float>& sum, vector<float>& out, float a)
	{
		(void)a;
		out = sum;
	}

	float derivate_single_identities(float out, float a)
	{
		(void)a;
		(void)out;
		return 1.f;
	}
	 /*Perceptron binary_step*/
	void activate_binary_step(vector<float>& sum, vector<float>& out, float a)
	{
		(void)a;
		if (out.size() != sum.size())
			out.resize(sum.size());

		for (unsigned int i = 0; i < sum.size(); i++)
		{
			if (sum[i] >= 0.)
				out[i] = 1.f;
			else
				out[i] = 0.f;
		}
	}

	float derivate_single_binary_step(float out, float a)
	{
		(void)a;
		if (out != 0.f)
			return 0.f;
		else
			return INFINITY;
	}

	/*Perceptron_logistic*/
	void activate_logistic(vector<float>& sum, vector<float>& out, float a)
	{
		(void)a;
		if (out.size() != sum.size())
			out.resize(sum.size());

		for(unsigned int i = 0; i < sum.size(); i++)
			out[i] = 1.f / (1.f + exp(-sum[i]));
	}

	float derivate_single_logistic(float out, float a)
	{
		(void)a;
		return out * (1.f - out);
	}

	/*Perceptron_tanH*/
	void activate_tanH(vector<float>& sum, vector<float>& out, float a)
	{
		(void)a;
		if (out.size() != sum.size())
			out.resize(sum.size());

		for(unsigned int i = 0; i < sum.size(); i++)
			out[i] = (2.f / (1.f + exp(-2.f * sum[i]))) - 1.f;
	}

	float derivate_single_tanH(float out, float a)
	{
		(void)a;
		return (1.f - out* out);
	}

	/*Perceptron arcTan*/
	void activate_arcTan(vector<float>& sum, vector<float>& out, float a)
	{
		(void)a;
		if (out.size() != sum.size())
			out.resize(sum.size());

		for(unsigned int i =0; i < sum.size(); i++)
			out[i] = atan(sum[i]);
	}

	float derivate_single_arcTan(float out, float a)
	{
		(void)a;
		return 1.f / (out *out + 1.f);
	}

	/*Perceptron_softsign*/
	void activate_softsign(vector<float>& sum, vector<float>& out, float a)
	{
		(void)a;
		if(out.size() != sum.size())
			out.resize(sum.size());

		for(unsigned int i = 0; i < sum.size(); i++)
			out[i] = sum[i] / (1.f + abs(sum[i]));
	}

	float derivate_single_softsign(float out, float a)
	{
		(void)a;
		return  1.f / ((abs(out) + 1.f)*(abs(out) + 1.f));
	}

	/*Perceptron rectifier*/
	void activate_rectifier(vector<float>& sum, vector<float>& out, float a)
	{
		(void)a;
		if (out.size() != sum.size())
			out.resize(sum.size());

		for (unsigned int i = 0; i < sum.size(); i++)
		{
			if (sum[i] >= 0.f)
				out[i] = sum[i];
			else
				out[i] = 0.f;
		}
	}

	float derivate_single_rectifier(float out, float a)
	{
		(void)a;
		if (out < 0.f)
			return 0.f;
		else
			return 1.;
	}

	/*Perceptron rectifier_param*/
	void activate_rectifier_param(vector<float>& sum, vector<float>& out, float a)
	{
		if (out.size() != sum.size())
			out.resize(sum.size());

		for (unsigned int i = 0; i < sum.size(); i++)
		{
			if (sum[i] >= 0.f)
				out[i] = sum[i];
			else
				out[i] = a*sum[i];
		}
	}

	float derivate_single_rectifier_param(float out, float a)
	{
		if (out < 0.f)
			return a;
		else
			return 1.f;
	}

	/*Perceptron_ELU*/
	void activate_ELU(vector<float>& sum, vector<float>& out, float a)
	{
		if (out.size() != sum.size())
			out.resize(sum.size());

		for (unsigned int i = 0; i < sum.size(); i++)
		{
			if (sum[i] >= 0.f)
				out[i] = sum[i];
			else
				out[i] = a*(exp(sum[i]) - 1.f);
		}
	}

	float derivate_single_ELU(float out, float a)
	{
		if (out < 0.f)
			return a + out;
		else
			return 1.f;
	}

	/*Perceptron_softPlus*/
	void activate_softPlus(vector<float>& sum, vector<float>& out, float a)
	{
		(void)a;
		if (out.size() != sum.size())
			out.resize(sum.size());

		for(unsigned int i = 0; i < sum.size(); i++)
			out[i] = log(1.f + exp(sum[i]));
	}

	float derivate_single_softPlus(float out, float a)
	{
		(void)a;
		return (float)(1.f / (1.f + exp(-out)));
	}

	/*Perceptron bent_identity*/
	void activate_bent_identity(vector<float>& sum, vector<float>& out, float a)
	{
		(void)a;
		if (out.size() != sum.size())
			out.resize(sum.size());

		for(unsigned int i = 0; i < sum.size(); i++)
			out[i] = (sqrt(sum[i]* sum[i] + 1.f) - 1.f) / 2.f + sum[i];
	}

	float derivate_single_bent_identity(float out, float a)
	{
		(void)a;
		return (float)(out / (2.f * sqrt(out * out + 1.f)) + 1.f);
	}

	/*Perceptron sinusoid*/
	void activate_sinusoid(vector<float>& sum, vector<float>& out, float a)
	{
		(void)a;
		if (out.size() != sum.size())
			out.resize(sum.size());

		for(unsigned int i = 0; i < sum.size(); i++)
			out[i] = sin(sum[i]);
	}

	float derivate_single_sinusoid(float out, float a)
	{
		(void)a;
		return cos(out);
	}

	/*Perceptron sinc*/
	void activate_sinc(vector<float>& sum, vector<float>& out, float a)
	{
		(void)a;
		if (out.size() != sum.size())
			out.resize(sum.size());

		for (unsigned int i = 0; i < sum.size(); i++)
		{
			if (sum[i] == 0.f)
				out[i] = 1.f;
			else
				out[i] = sin(sum[i]) / sum[i];
		}
	}

	float derivate_single_sinc(float out, float a)
	{
		(void)a;
		if (out == 0.f)
			return 0.f;
		else
			return cos(out) / out + sin(out) / (out * out);
	}

	/*Perceptron gaussian*/
	void activate_gaussian(vector<float>& sum, vector<float>& out, float a)
	{
		(void)a;
		if (out.size() != sum.size())
			out.resize(sum.size());

		for(unsigned int i = 0; i < sum.size(); i++)
			out[i] = exp(-sum[i]*sum[i]);
	}

	float derivate_single_gaussian(float out, float a)
	{
		(void)a;
		return (float)(-2.f * exp(-out* out));
	}

} // namespace SNN_network
