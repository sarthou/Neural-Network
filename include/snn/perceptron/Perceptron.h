//============================================================================
// Name        : Perceptron.h
// Authors     : Guillaume Sarthou
// EMail       : open.pode@gmail.com
// Date		   : 29 jun. 2017
// Version     : V1.5
// Copyright   : This file is part of SNN_network project which is released under
//               MIT license.
//============================================================================

#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include <vector>
#include <string>

namespace SNN
{
	using namespace std;

	enum perceptron_type_t
	{
		identities,
		binary_step,
		logistic,
		tanH,
		arcTan,
		softsign,
		rectifier,
		rectifier_param,
		ELU,
		softPlus,
		bent_identity,
		sinusoid,
		sinc,
		gaussian,
		input
	};

	class Trainig_process;

	class Perceptron
	{
		friend class Trainig_process;
		friend class Trainer;
	public:
		Perceptron(int p_layer, int p_id, perceptron_type_t type = identities, float param = 0);
		Perceptron(Perceptron const& perceptron);
		virtual ~Perceptron();

		Perceptron& operator=(Perceptron const& perceptron);

		void set_input(vector<Perceptron*>* p_input_perceptrons);
		virtual void set_input(const vector<vector<float>*>& p_input) { (void)p_input; };
		virtual void set_input(const vector<float>& p_input) { (void)p_input; };

		void set_weigh(const vector<float>& p_w);
		inline vector<float> get_weigh() { return m_w; };
		void set_bia(float p_bia) { m_bia = p_bia; };
		inline float get_bia() { return m_bia; };

		inline vector<float>* get_output() { return &m_out; };
		inline vector<float> get_output_cpy() { return m_out; };
		inline void clr() { m_out.clear(); m_sum.clear(); m_derivate.clear(); };

		inline void activate()
		{
			sum();
			(*activate_ptr)(m_sum, m_out, m_a);
		}
		inline string get_type()
		{
			return (*get_type_ptr)();
		}

	protected:
		int m_layer;
		int m_id;
		perceptron_type_t m_type;

		float m_bia;
		vector<float> m_w;
		float m_a;

		vector<Perceptron*>* m_input_perceptrons;
		vector<float> m_out;

		vector<float> m_sum;
		vector<float> m_derivate;

		inline void sum()
		{
			if (m_w.size() > 0)
			{
				unsigned int size = (*m_input_perceptrons)[0]->m_out.size();
				m_sum.resize(size);
				for (unsigned int i = 0; i < size; i++)
					m_sum[i] = -m_bia;

				Perceptron* perceptron = nullptr;
				size = (*m_input_perceptrons)[0]->m_out.size();
				float w = 0;
				for (unsigned int vect = 0; vect < m_input_perceptrons->size(); vect++)
				{
					perceptron = (*m_input_perceptrons)[vect];
					w = m_w[vect];
					for (unsigned int i = 0; i < size; i++)
						m_sum[i] += w * perceptron->m_out[i];
				}
			}
		}

		void set_functions();
		void (*activate_ptr)(vector<float>&, vector<float>&, float);
		string(*get_type_ptr)(void);

		float (*derivate_single_ptr)(float, float);
		inline float derivate_single()
		{
			return (*derivate_single_ptr)(m_out[0], m_a);
		}

	};

} // namespace SNN_network

#endif