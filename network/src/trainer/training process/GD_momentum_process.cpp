#include "GD_momentum_process.h"

namespace SNN_network
{

	GD_momentum_process::GD_momentum_process(Perceptron* p_perceptron, double p_step) : Trainig_process(p_perceptron)
	{
		m_step = p_step;
		m_delta_1 = 0;
		m_delta = 0;
		m_error = 0;
		m_past_time_update = 0.9;
	}

	GD_momentum_process::~GD_momentum_process()
	{

	}

	void GD_momentum_process::init()
	{
		m_w_gradient.resize(m_perceptron->get_weigh().size());
		m_delta_1 = m_delta;
		m_delta = 0;
		m_error = 0;
		m_past_time_update = 0.9;
	}

	void GD_momentum_process::set_error(double T)
	{
		if (1 != m_perceptron->get_output()->size())
			cout << "Internal training error" << endl;
		else
		{
			vector<double>::iterator out_it = m_perceptron->get_output()->begin();
			m_error = (*out_it) - T;
		}
	}

	void GD_momentum_process::propagate(vector<Trainig_process*> process, bool out)
	{
		derivate_perceptron();

		if (out)
			m_delta = -m_error*get_derivate() + m_past_time_update*m_delta_1*m_step;
		else
			m_delta *= get_derivate() + m_past_time_update*m_delta_1*m_step;

		add_to_precedent(process, m_delta);
	}

	void GD_momentum_process::compute()
	{
		m_bia_gradient = m_delta;
		m_perceptron->set_bia(m_perceptron->get_bia() - m_bia_gradient*m_step);

		vector<double> in = get_inputs();
		if (in.size() == m_w_gradient.size())
		{
			vector<double> w = m_perceptron->get_weigh();
			vector<double>::iterator it_w = w.begin();
			for (vector<double>::iterator it = in.begin(); it != in.end(); ++it)
			{
				(*it_w) += (*it)*m_delta*m_step;
				it_w++;
			}

			m_perceptron->set_weigh(w);
		}
	}

	void GD_momentum_process::add(double value)
	{
		m_delta += value;
	}

} // namespace SNN_trainer