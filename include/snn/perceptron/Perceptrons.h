//============================================================================
// Name        : Perceptrons.h
// Authors     : Guillaume Sarthou
// EMail       : open.pode@gmail.com
// Date		   : 18 jul. 2017
// Version     : V2.0
// Copyright   : This file is part of SNN_network project which is released under
//               MIT license.
//============================================================================
#ifndef PERCEPTRONS_H
#define PERCEPTRONS_H

#include <string>
#include <vector>

namespace SNN
{
	using namespace std;

	/*activate functions*/
	void activate_identities(vector<float>& sum, vector<float>& out, float a);
	void activate_binary_step(vector<float>& sum, vector<float>& out, float a);
	void activate_logistic(vector<float>& sum, vector<float>& out, float a);
	void activate_tanH(vector<float>& sum, vector<float>& out, float a);
	void activate_arcTan(vector<float>& sum, vector<float>& out, float a);
	void activate_softsign(vector<float>& sum, vector<float>& out, float a);
	void activate_rectifier(vector<float>& sum, vector<float>& out, float a);
	void activate_rectifier_param(vector<float>& sum, vector<float>& out, float a);
	void activate_ELU(vector<float>& sum, vector<float>& out, float a);
	void activate_softPlus(vector<float>& sum, vector<float>& out, float a);
	void activate_bent_identity(vector<float>& sum, vector<float>& out, float a);
	void activate_sinusoid(vector<float>& sum, vector<float>& out, float a);
	void activate_sinc(vector<float>& sum, vector<float>& out, float a);
	void activate_gaussian(vector<float>& sum, vector<float>& out, float a);
	inline void activate_input(vector<float>& sum, vector<float>& out, float a) { (void)out, (void)sum; (void)a; }

	/*derivate_single functions*/
	float derivate_single_identities(float out, float a);
	float derivate_single_binary_step(float out, float a);
	float derivate_single_logistic(float out, float a);
	float derivate_single_tanH(float out, float a);
	float derivate_single_arcTan(float out, float a);
	float derivate_single_softsign(float out, float a);
	float derivate_single_rectifier(float out, float a);
	float derivate_single_rectifier_param(float out, float a);
	float derivate_single_ELU(float out, float a);
	float derivate_single_softPlus(float out, float a);
	float derivate_single_bent_identity(float out, float a);
	float derivate_single_sinusoid(float out, float a);
	float derivate_single_sinc(float out, float a);
	float derivate_single_gaussian(float out, float a);
	inline float derivate_single_input(float out, float a) { (void)out, (void)a; return INFINITY; }

	/*get_type functions*/
	inline string get_type_identities() { return "identity"; }
	inline string get_type_binary_step() { return "binary_step"; }
	inline string get_type_logistic() { return "logistic"; }
	inline string get_type_tanH() { return "tanH"; }
	inline string get_type_arcTan() { return "arcTan"; }
	inline string get_type_softsign() { return "softsign"; }
	inline string get_type_rectifier() { return "rectifier"; }
	inline string get_type_rectifier_param() { return "rectifier_param"; }
	inline string get_type_ELU() { return "ELU"; }
	inline string get_type_softPlus() { return "softPlus"; }
	inline string get_type_bent_identity() { return "bent_identity"; }
	inline string get_type_sinusoid() { return "sinusoid"; }
	inline string get_type_sinc() { return "sinc"; }
	inline string get_type_gaussian() { return "gaussian"; }
	inline string get_type_input() { return "input"; }

}

#endif
