#pragma once

#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include <vector>

class Trainig_process;

using namespace std;

class Perceptron
{
	friend class Trainig_process;
	friend class Trainer;
public:
	Perceptron(int p_layer, int p_id);
	virtual ~Perceptron();

	void set_input(vector<Perceptron*>* p_input_perceptrons);
	bool set_input(vector<vector<double>*> p_input);

	void set_weigh(vector<double> p_w);
	vector<double> get_weigh() { return m_w; };
	void set_bia(double p_bia) { m_bia = p_bia; };
	double get_bia() { return m_bia; };

	vector<double>* get_output() { return &m_out; };
	vector<double> get_output_cpy() { return m_out; };
	void clr() { m_out.clear(); m_sum.clear(); m_derivate.clear(); };

	virtual void activate() = 0;
	virtual string get_type() = 0;

protected:
	int m_layer;
	int m_id;

	double m_bia;
	vector<double> m_w;

	vector<Perceptron*>* m_input_perceptrons;
	vector<vector<double>*> m_in;
	vector<double> m_out;

	vector<double> m_sum;
	vector<double> m_derivate;

	void sum();
	virtual void derivate() = 0;
};

#endif
