/*********************************
*
*	Sarthou Guillaume
*	30/11/2016
*
*********************************
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

class Perceptron_identity : public Perceptron
{
public:
	Perceptron_identity(int p_layer, int p_id) : Perceptron(p_layer, p_id) {};
	~Perceptron_identity() {};

	void activate();
	string get_type() { return "identity"; };
private:
	void derivate();
};

class Perceptron_binary_step : public Perceptron // {0,1}
{
public:
	Perceptron_binary_step(int p_layer, int p_id) : Perceptron(p_layer, p_id) {};
	~Perceptron_binary_step() {};

	void activate();
	string get_type() { return "binary_step"; };
private:
	void derivate();
};

class Perceptron_logistic : public Perceptron // (0,1)
{
public:
	Perceptron_logistic(int p_layer, int p_id) : Perceptron(p_layer, p_id) {};
	~Perceptron_logistic() {};

	void activate();
	string get_type() { return "logistic"; };
private:
	void derivate();
};

class Perceptron_tanH : public Perceptron // (-1,1)
{
public:
	Perceptron_tanH(int p_layer, int p_id) : Perceptron(p_layer, p_id) {};
	~Perceptron_tanH() {};

	void activate();
	string get_type() { return "tanH"; };
private:
	void derivate();
};

class Perceptron_arcTan : public Perceptron // (-pi/2, pi/2)
{
public:
	Perceptron_arcTan(int p_layer, int p_id) : Perceptron(p_layer, p_id) {};
	~Perceptron_arcTan() {};

	void activate();
	string get_type() { return "arcTan"; };
private:
	void derivate();
};

class Perceptron_softsign : public Perceptron // (-1, 1)
{
public:
	Perceptron_softsign(int p_layer, int p_id) : Perceptron(p_layer, p_id) {};
	~Perceptron_softsign() {};

	void activate();
	string get_type() { return "softsign"; };
private:
	void derivate();
};

class Perceptron_rectifier : public Perceptron // [0, inf)
{
public:
	Perceptron_rectifier(int p_layer, int p_id) : Perceptron(p_layer, p_id) {};
	~Perceptron_rectifier() {};

	void activate();
	string get_type() { return "rectifier"; };
private:
	void derivate();
};

class Perceptron_rectifier_param : public Perceptron // (-inf, inf)
{
public:
	Perceptron_rectifier_param(int p_layer, int p_id, double a = 0.) : Perceptron(p_layer, p_id) { m_a = a; };
	~Perceptron_rectifier_param() {};

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
	~Perceptron_ELU() {};

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
	~Perceptron_softPlus() {};

	void activate();
	string get_type() { return "softPlus"; };
private:
	void derivate();
};

class Perceptron_bent_identity : public Perceptron // (-inf, inf)
{
public:
	Perceptron_bent_identity(int p_layer, int p_id) : Perceptron(p_layer, p_id) {};
	~Perceptron_bent_identity() {};

	void activate();
	string get_type() { return "bent_identity"; };
private:
	void derivate();
};

class Perceptron_sinusoid : public Perceptron // (-1, 1)
{
public:
	Perceptron_sinusoid(int p_layer, int p_id) : Perceptron(p_layer, p_id) {};
	~Perceptron_sinusoid() {};

	void activate();
	string get_type() { return "sinusoid"; };
private:
	void derivate();
};

class Perceptron_sinc : public Perceptron // (-0.217, 1)
{
public:
	Perceptron_sinc(int p_layer, int p_id) : Perceptron(p_layer, p_id) {};
	~Perceptron_sinc() {};

	void activate();
	string get_type() { return "sinc"; };
private:
	void derivate();
};

class Perceptron_gaussian : public Perceptron // (0, 1]
{
public:
	Perceptron_gaussian(int p_layer, int p_id) : Perceptron(p_layer, p_id) {};
	~Perceptron_gaussian() {};

	void activate();
	string get_type() { return "gaussian"; };
private:
	void derivate();
};

#endif
