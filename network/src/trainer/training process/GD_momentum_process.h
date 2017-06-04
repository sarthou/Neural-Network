#pragma once

#ifndef GD_MOMENTUM_PROCESS_H
#define GD_MOMENTUM_PROCESS_H

#include "Trainig_process.h"

namespace SNN_network
{

	class GD_momentum_process : public Trainig_process
	{
	public:
		GD_momentum_process(Perceptron* p_perceptron, double p_step);
		~GD_momentum_process();

		void init();
		void set_error(double T);
		void propagate(vector<Trainig_process*> process, bool out = false);
		void compute();

		void add(double value);

	private:
		double m_step;
		double m_past_time_update;
		double m_bia_gradient;
		vector<double> m_w_gradient;

		double m_error;
		double m_delta;
		double m_delta_1;
	};
} // SNN_network

#endif // !GD_MOMENTUM_PROCESS_H
