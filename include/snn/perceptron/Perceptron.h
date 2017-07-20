//============================================================================
// Name        : Perceptron.h
// Authors     : Guillaume Sarthou
// EMail       : open.pode@gmail.com
// Date		   : 18 jul. 2017
// Version     : V2.0
// Copyright   : This file is part of SNN_network project which is released under
//               MIT license.
//============================================================================

#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include <vector>
#include <string>
#include <cmath>

namespace SNN
{
	using namespace std;

	enum perceptron_type_t
	{
		identities,			// (-inf, inf)
		binary_step,		// {0,1}
		logistic,			// (0,1)
		tanH,				// (-1,1)
		arcTan,				// (-pi/2, pi/2)
		softsign,			// (-1, 1)
		rectifier,			// [0, inf)
		rectifier_param,	// (-inf, inf)
		ELU,				// (-a, inf)
		softPlus,			// (0, inf)
		bent_identity,		// (-inf, inf)
		sinusoid,			// (-1, 1)
		sinc,				// (-0.217, 1)
		gaussian,			// (0, 1]
		input				// don't use
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