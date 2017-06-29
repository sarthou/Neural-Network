//============================================================================
// Name        : Perceptrons.h
// Authors     : Guillaume Sarthou
// EMail       : open.pode@gmail.com
// Date		   : 29 jun. 2016
// Version     : V1.5
// Copyright   : This file is part of SNN_network project which is released under
//               MIT license.
//============================================================================

/*
* Perceptron_identity
* Perceptron_binary_step
* Perceptron_logistic
* Perceptron_tanH
* Perceptron_arcTan
* Perceptron_softsign
* Perceptron_rectifier
* Perceptron_rectifier_param
* Perceptron_ELU
* Perceptron_softPlus
* Perceptron_bent_identity
* Perceptron_sinusoid
* Perceptron_sinc
* Perceptron_gaussian
*/

#include "Perceptron.h"

#ifndef PERCEPTRONS_H
#define PERCEPTRONS_H

namespace SNN
{

	class Perceptron_identity : public Perceptron
	{
	public:
		Perceptron_identity(int p_layer, int p_id) : Perceptron(p_layer, p_id) {};
		Perceptron_identity(Perceptron_identity const& perceptron) : Perceptron(perceptron) {};
		~Perceptron_identity() {};

		Perceptron_identity& operator=(Perceptron_identity const& perceptron) { Perceptron::operator=(perceptron); };

		void activate();
		string get_type() { return "identity"; };
	private:
		void derivate();
	};

	class Perceptron_binary_step : public Perceptron // {0,1}
	{
	public:
		Perceptron_binary_step(int p_layer, int p_id) : Perceptron(p_layer, p_id) {};
		Perceptron_binary_step(Perceptron_binary_step const& perceptron) : Perceptron(perceptron) {};
		~Perceptron_binary_step() {};

		Perceptron_binary_step& operator=(Perceptron_binary_step const& perceptron) { Perceptron::operator=(perceptron); };

		void activate();
		string get_type() { return "binary_step"; };
	private:
		void derivate();
	};

	class Perceptron_logistic : public Perceptron // (0,1)
	{
	public:
		Perceptron_logistic(int p_layer, int p_id) : Perceptron(p_layer, p_id) {};
		Perceptron_logistic(Perceptron_logistic const& perceptron) : Perceptron(perceptron) {};
		~Perceptron_logistic() {};

		Perceptron_logistic& operator=(Perceptron_logistic const& perceptron) { Perceptron::operator=(perceptron); };

		void activate();
		string get_type() { return "logistic"; };
	private:
		void derivate();
	};

	class Perceptron_tanH : public Perceptron // (-1,1)
	{
	public:
		Perceptron_tanH(int p_layer, int p_id) : Perceptron(p_layer, p_id) {};
		Perceptron_tanH(Perceptron_tanH const& perceptron) : Perceptron(perceptron) {};
		~Perceptron_tanH() {};

		Perceptron_tanH& operator=(Perceptron_tanH const& perceptron) { Perceptron::operator=(perceptron); };

		void activate();
		string get_type() { return "tanH"; };
	private:
		void derivate();
	};

	class Perceptron_arcTan : public Perceptron // (-pi/2, pi/2)
	{
	public:
		Perceptron_arcTan(int p_layer, int p_id) : Perceptron(p_layer, p_id) {};
		Perceptron_arcTan(Perceptron_arcTan const& perceptron) : Perceptron(perceptron) {};
		~Perceptron_arcTan() {};

		Perceptron_arcTan& operator=(Perceptron_arcTan const& perceptron) { Perceptron::operator=(perceptron); };

		void activate();
		string get_type() { return "arcTan"; };
	private:
		void derivate();
	};

	class Perceptron_softsign : public Perceptron // (-1, 1)
	{
	public:
		Perceptron_softsign(int p_layer, int p_id) : Perceptron(p_layer, p_id) {};
		Perceptron_softsign(Perceptron_softsign const& perceptron) : Perceptron(perceptron) {};
		~Perceptron_softsign() {};

		Perceptron_softsign& operator=(Perceptron_softsign const& perceptron) { Perceptron::operator=(perceptron); };

		void activate();
		string get_type() { return "softsign"; };
	private:
		void derivate();
	};

	class Perceptron_rectifier : public Perceptron // [0, inf)
	{
	public:
		Perceptron_rectifier(int p_layer, int p_id) : Perceptron(p_layer, p_id) {};
		Perceptron_rectifier(Perceptron_rectifier const& perceptron) : Perceptron(perceptron) {};
		~Perceptron_rectifier() {};

		Perceptron_rectifier& operator=(Perceptron_rectifier const& perceptron) { Perceptron::operator=(perceptron); };

		void activate();
		string get_type() { return "rectifier"; };
	private:
		void derivate();
	};

	class Perceptron_rectifier_param : public Perceptron // (-inf, inf)
	{
	public:
		Perceptron_rectifier_param(int p_layer, int p_id, double a = 0.) : Perceptron(p_layer, p_id) { m_a = a; };
		Perceptron_rectifier_param(Perceptron_rectifier_param const& perceptron) : Perceptron(perceptron) { m_a = perceptron.m_a; };
		~Perceptron_rectifier_param() {};

		Perceptron_rectifier_param& operator=(Perceptron_rectifier_param const& perceptron) { Perceptron::operator=(perceptron); m_a = perceptron.m_a; };

		void activate();
		string get_type() { return "rectifier_param"; };
	private:
		double m_a;

		void derivate();
	};

	class Perceptron_ELU : public Perceptron // (-a, inf) Exponential Linear Unit
	{
	public:
		Perceptron_ELU(int p_layer, int p_id, double a = 0.) : Perceptron(p_layer, p_id) { m_a = a; };
		Perceptron_ELU(Perceptron_ELU const& perceptron) : Perceptron(perceptron) { m_a = perceptron.m_a; };
		~Perceptron_ELU() {};

		Perceptron_ELU& operator=(Perceptron_ELU const& perceptron) { Perceptron::operator=(perceptron); m_a = perceptron.m_a; };

		void activate();
		string get_type() { return "ELU"; };
	private:
		double m_a;

		void derivate();
	};

	class Perceptron_softPlus : public Perceptron // (0, inf)
	{
	public:
		Perceptron_softPlus(int p_layer, int p_id) : Perceptron(p_layer, p_id) {};
		Perceptron_softPlus(Perceptron_softPlus const& perceptron) : Perceptron(perceptron) {};
		~Perceptron_softPlus() {};

		Perceptron_softPlus& operator=(Perceptron_softPlus const& perceptron) { Perceptron::operator=(perceptron); };

		void activate();
		string get_type() { return "softPlus"; };
	private:
		void derivate();
	};

	class Perceptron_bent_identity : public Perceptron // (-inf, inf)
	{
	public:
		Perceptron_bent_identity(int p_layer, int p_id) : Perceptron(p_layer, p_id) {};
		Perceptron_bent_identity(Perceptron_bent_identity const& perceptron) : Perceptron(perceptron) {};
		~Perceptron_bent_identity() {};

		Perceptron_bent_identity& operator=(Perceptron_bent_identity const& perceptron) { Perceptron::operator=(perceptron); };

		void activate();
		string get_type() { return "bent_identity"; };
	private:
		void derivate();
	};

	class Perceptron_sinusoid : public Perceptron // (-1, 1)
	{
	public:
		Perceptron_sinusoid(int p_layer, int p_id) : Perceptron(p_layer, p_id) {};
		Perceptron_sinusoid(Perceptron_sinusoid const& perceptron) : Perceptron(perceptron) {};
		~Perceptron_sinusoid() {};

		Perceptron_sinusoid& operator=(Perceptron_sinusoid const& perceptron) { Perceptron::operator=(perceptron); };

		void activate();
		string get_type() { return "sinusoid"; };
	private:
		void derivate();
	};

	class Perceptron_sinc : public Perceptron // (-0.217, 1)
	{
	public:
		Perceptron_sinc(int p_layer, int p_id) : Perceptron(p_layer, p_id) {};
		Perceptron_sinc(Perceptron_sinc const& perceptron) : Perceptron(perceptron) {};
		~Perceptron_sinc() {};

		Perceptron_sinc& operator=(Perceptron_sinc const& perceptron) { Perceptron::operator=(perceptron); };

		void activate();
		string get_type() { return "sinc"; };
	private:
		void derivate();
	};

	class Perceptron_gaussian : public Perceptron // (0, 1]
	{
	public:
		Perceptron_gaussian(int p_layer, int p_id) : Perceptron(p_layer, p_id) {};
		Perceptron_gaussian(Perceptron_gaussian const& perceptron) : Perceptron(perceptron) {};
		~Perceptron_gaussian() {};

		Perceptron_gaussian& operator=(Perceptron_gaussian const& perceptron) { Perceptron::operator=(perceptron); };

		void activate();
		string get_type() { return "gaussian"; };
	private:
		void derivate();
	};

} //namespace SNN_network

#endif
