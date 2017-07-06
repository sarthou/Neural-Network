//============================================================================
// Name        : Perceptrons.cpp
// Authors     : Guillaume Sarthou
// EMail       : open.pode@gmail.com
// Date		   : 29 jun. 2017
// Version     : V1.5
// Copyright   : This file is part of SNN_network project which is released under
//               MIT license.
//============================================================================
#include "snn/perceptron/Perceptrons.h"

#include <cmath>

namespace SNN
{

	void Perceptron_identity::activate()
	{
		sum();
		m_out = m_sum;
	}

	void Perceptron_identity::derivate()
	{
		m_derivate = m_sum;

		vector<float>::iterator it_deriv = m_derivate.begin();
		for (vector<float>::iterator it = m_sum.begin(); it != m_sum.end(); ++it)
		{
			(*it_deriv) = 1.f;

			++it_deriv;
		}
	}

	float Perceptron_identity::derivate_single()
	{
		return 1.f;
	}

	void Perceptron_binary_step::activate()
	{
		sum();
		m_out = m_sum;

		vector<float>::iterator it_out = m_out.begin();
		for (vector<float>::iterator it = m_sum.begin(); it != m_sum.end(); ++it)
		{
			if ((*it) >= 0.)
				(*it_out) = 1.f;
			else
				(*it_out) = 0.f;

			++it_out;
		}
	}

	void Perceptron_binary_step::derivate()
	{
		m_derivate = m_sum;

		vector<float>::iterator it_deriv = m_derivate.begin();
		for (vector<float>::iterator it = m_sum.begin(); it != m_sum.end(); ++it)
		{
			if ((*it) != 0.f)
				(*it_deriv) = 0.f;
			else
				(*it_deriv) = INFINITY;

			++it_deriv;
		}
	}

	float Perceptron_binary_step::derivate_single()
	{
		if (m_sum[0] != 0.f)
			return 0.f;
		else
			return INFINITY;
	}

	void Perceptron_logistic::activate()
	{
		sum();
		m_out = m_sum;

		vector<float>::iterator it_out = m_out.begin();
		for (vector<float>::iterator it = m_sum.begin(); it != m_sum.end(); ++it)
		{
			(*it_out) = 1.f / (1.f + exp(-(*it)));

			++it_out;
		}
	}

	void Perceptron_logistic::derivate()
	{
		m_derivate = m_sum;

		vector<float>::iterator it_deriv = m_derivate.begin();
		for (vector<float>::iterator it = m_out.begin(); it != m_out.end(); ++it)
		{
			(*it_deriv) = (*it)*(1.f - (*it));

			++it_deriv;
		}
	}

	float Perceptron_logistic::derivate_single()
	{
		return m_out[0] * (1.f - m_out[0]);
	}

	void Perceptron_tanH::activate()
	{
		sum();
		m_out = m_sum;

		vector<float>::iterator it_out = m_out.begin();
		for (vector<float>::iterator it = m_sum.begin(); it != m_sum.end(); ++it)
		{
			(*it_out) = (2.f / (1.f + exp(-2.f * (*it)))) - 1.f;

			++it_out;
		}
	}

	void Perceptron_tanH::derivate()
	{
		m_derivate = m_sum;

		vector<float>::iterator it_deriv = m_derivate.begin();
		for (vector<float>::iterator it = m_out.begin(); it != m_out.end(); ++it)
		{
			(*it_deriv) = 1.f - (*it)*(*it);

			++it_deriv;
		}
	}

	float Perceptron_tanH::derivate_single()
	{
		return (1.f - m_out[0]* m_out[0]);
	}

	void Perceptron_arcTan::activate()
	{
		sum();
		m_out = m_sum;

		vector<float>::iterator it_out = m_out.begin();
		for (vector<float>::iterator it = m_sum.begin(); it != m_sum.end(); ++it)
		{
			(*it_out) = atan(*it);

			++it_out;
		}
	}

	void Perceptron_arcTan::derivate()
	{
		m_derivate = m_sum;

		vector<float>::iterator it_deriv = m_derivate.begin();
		for (vector<float>::iterator it = m_sum.begin(); it != m_sum.end(); ++it)
		{
			(*it_deriv) = 1.f / ((*it)*(*it) + 1.f);

			++it_deriv;
		}
	}

	float Perceptron_arcTan::derivate_single()
	{
		return 1.f / (m_out[0] *m_out[0] + 1.f);
	}

	void Perceptron_softsign::activate()
	{
		sum();
		m_out = m_sum;

		vector<float>::iterator it_out = m_out.begin();
		for (vector<float>::iterator it = m_sum.begin(); it != m_sum.end(); ++it)
		{
			(*it_out) = (*it) / (1.f + abs(*it));

			++it_out;
		}
	}

	void Perceptron_softsign::derivate()
	{
		m_derivate = m_sum;

		vector<float>::iterator it_deriv = m_derivate.begin();
		for (vector<float>::iterator it = m_sum.begin(); it != m_sum.end(); ++it)
		{
			(*it_deriv) = 1.f / ((abs((*it)) + 1.f)*(abs((*it)) + 1.f));

			++it_deriv;
		}
	}

	float Perceptron_softsign::derivate_single()
	{
		return  1.f / ((abs(m_out[0]) + 1.f)*(abs(m_out[0]) + 1.f));
	}

	void Perceptron_rectifier::activate()
	{
		sum();
		m_out = m_sum;

		vector<float>::iterator it_out = m_out.begin();
		for (vector<float>::iterator it = m_sum.begin(); it != m_sum.end(); ++it)
		{
			if ((*it) >= 0.f)
				(*it_out) = (*it);
			else
				(*it_out) = 0.f;

			++it_out;
		}
	}

	void Perceptron_rectifier::derivate()
	{
		m_derivate = m_sum;

		vector<float>::iterator it_deriv = m_derivate.begin();
		for (vector<float>::iterator it = m_sum.begin(); it != m_sum.end(); ++it)
		{
			if ((*it) < 0.f)
				(*it_deriv) = 0.f;
			else
				(*it_deriv) = 1.f;

			++it_deriv;
		}
	}

	float Perceptron_rectifier::derivate_single()
	{
		if (m_out[0] < 0.f)
			return 0.f;
		else
			return 1.;
	}

	void Perceptron_rectifier_param::activate()
	{
		sum();
		m_out = m_sum;

		vector<float>::iterator it_out = m_out.begin();
		for (vector<float>::iterator it = m_sum.begin(); it != m_sum.end(); ++it)
		{
			if ((*it) >= 0.f)
				(*it_out) = (*it);
			else
				(*it_out) = m_a*(*it);

			++it_out;
		}
	}

	void Perceptron_rectifier_param::derivate()
	{
		m_derivate = m_sum;

		vector<float>::iterator it_deriv = m_derivate.begin();
		for (vector<float>::iterator it = m_sum.begin(); it != m_sum.end(); ++it)
		{
			if ((*it) < 0.f)
				(*it_deriv) = m_a;
			else
				(*it_deriv) = 1.f;

			++it_deriv;
		}
	}

	float Perceptron_rectifier_param::derivate_single()
	{
		if (m_out[0] < 0.f)
			return m_a;
		else
			return 1.f;
	}

	void Perceptron_ELU::activate()
	{
		sum();
		m_out = m_sum;

		vector<float>::iterator it_out = m_out.begin();
		for (vector<float>::iterator it = m_sum.begin(); it != m_sum.end(); ++it)
		{
			if ((*it) >= 0.f)
				(*it_out) = (*it);
			else
				(*it_out) = m_a*(exp(*it) - 1.f);

			++it_out;
		}
	}

	void Perceptron_ELU::derivate()
	{
		m_derivate = m_sum;

		vector<float>::iterator it_deriv = m_derivate.begin();
		vector<float>::iterator it_out = m_out.begin();
		for (vector<float>::iterator it = m_sum.begin(); it != m_sum.end(); ++it)
		{
			if ((*it) < 0.f)
				(*it_deriv) = m_a + (*it_out);
			else
				(*it_deriv) = 1.f;

			++it_deriv;
			++it_out;
		}
	}

	float Perceptron_ELU::derivate_single()
	{
		if (m_out[0] < 0.f)
			return m_a + m_out[0];
		else
			return 1.f;
	}

	void Perceptron_softPlus::activate()
	{
		sum();
		m_out = m_sum;

		vector<float>::iterator it_out = m_out.begin();
		for (vector<float>::iterator it = m_sum.begin(); it != m_sum.end(); ++it)
		{
			(*it_out) = log(1.f + exp(*it));

			++it_out;
		}
	}

	void Perceptron_softPlus::derivate()
	{
		m_derivate = m_sum;

		vector<float>::iterator it_deriv = m_derivate.begin();
		for (vector<float>::iterator it = m_sum.begin(); it != m_sum.end(); ++it)
		{
			(*it_deriv) = 1.f / (1.f + exp(-(*it)));

			++it_deriv;
		}
	}

	float Perceptron_softPlus::derivate_single()
	{
		return (float)(1.f / (1.f + exp(-m_out[0])));
	}

	void Perceptron_bent_identity::activate()
	{
		sum();
		m_out = m_sum;

		vector<float>::iterator it_out = m_out.begin();
		for (vector<float>::iterator it = m_sum.begin(); it != m_sum.end(); ++it)
		{
			(*it_out) = (sqrt((*it)*(*it) + 1.f) - 1.f) / 2.f + (*it);

			++it_out;
		}
	}

	void Perceptron_bent_identity::derivate()
	{
		m_derivate = m_sum;

		vector<float>::iterator it_deriv = m_derivate.begin();
		for (vector<float>::iterator it = m_sum.begin(); it != m_sum.end(); ++it)
		{
			(*it_deriv) = (*it) / (2.f * sqrt((*it)*(*it) + 1.f)) + 1.f;

			++it_deriv;
		}
	}

	float Perceptron_bent_identity::derivate_single()
	{
		return (float)(m_out[0] / (2.f * sqrt(m_out[0] * m_out[0] + 1.f)) + 1.f);
	}

	void Perceptron_sinusoid::activate()
	{
		sum();
		m_out = m_sum;

		vector<float>::iterator it_out = m_out.begin();
		for (vector<float>::iterator it = m_sum.begin(); it != m_sum.end(); ++it)
		{
			(*it_out) = sin(*it);

			++it_out;
		}
	}

	void Perceptron_sinusoid::derivate()
	{
		m_derivate = m_sum;

		vector<float>::iterator it_deriv = m_derivate.begin();
		for (vector<float>::iterator it = m_sum.begin(); it != m_sum.end(); ++it)
		{
			(*it_deriv) = cos(*it);

			++it_deriv;
		}
	}

	float Perceptron_sinusoid::derivate_single()
	{
		return cos(m_out[0]);
	}

	void Perceptron_sinc::activate()
	{
		sum();
		m_out = m_sum;

		vector<float>::iterator it_out = m_out.begin();
		for (vector<float>::iterator it = m_sum.begin(); it != m_sum.end(); ++it)
		{
			if ((*it) == 0.f)
				(*it_out) = 1.f;
			else
				(*it_out) = sin(*it) / (*it);

			++it_out;
		}
	}

	void Perceptron_sinc::derivate()
	{
		m_derivate = m_sum;

		vector<float>::iterator it_deriv = m_derivate.begin();
		for (vector<float>::iterator it = m_sum.begin(); it != m_sum.end(); ++it)
		{
			if ((*it) == 0.f)
				(*it_deriv) = 0.f;
			else
				(*it_deriv) = cos((*it)) / (*it) + sin((*it)) / ((*it)*(*it));

			++it_deriv;
		}
	}

	float Perceptron_sinc::derivate_single()
	{
		if (m_out[0] == 0.f)
			return 0.f;
		else
			return cos(m_out[0]) / m_out[0] + sin(m_out[0]) / (m_out[0] * m_out[0]);
	}

	void Perceptron_gaussian::activate()
	{
		sum();
		m_out = m_sum;

		vector<float>::iterator it_out = m_out.begin();
		for (vector<float>::iterator it = m_sum.begin(); it != m_sum.end(); ++it)
		{
			(*it_out) = exp(-(*it)*(*it));

			++it_out;
		}
	}

	void Perceptron_gaussian::derivate()
	{
		m_derivate = m_sum;

		vector<float>::iterator it_deriv = m_derivate.begin();
		for (vector<float>::iterator it = m_sum.begin(); it != m_sum.end(); ++it)
		{
			(*it_deriv) = -2.f * exp(-(*it)*(*it));

			++it_deriv;
		}
	}

	float Perceptron_gaussian::derivate_single()
	{
		return (float)(-2.f * exp(-m_out[0]* m_out[0]));
	}

} // namespace SNN_network
