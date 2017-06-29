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

		vector<double>::iterator it_deriv = m_derivate.begin();
		for (vector<double>::iterator it = m_sum.begin(); it != m_sum.end(); ++it)
		{
			(*it_deriv) = 1.;

			++it_deriv;
		}
	}

	void Perceptron_binary_step::activate()
	{
		sum();
		m_out = m_sum;

		vector<double>::iterator it_out = m_out.begin();
		for (vector<double>::iterator it = m_sum.begin(); it != m_sum.end(); ++it)
		{
			if ((*it) >= 0.)
				(*it_out) = 1.;
			else
				(*it_out) = 0.;

			++it_out;
		}
	}

	void Perceptron_binary_step::derivate()
	{
		m_derivate = m_sum;

		vector<double>::iterator it_deriv = m_derivate.begin();
		for (vector<double>::iterator it = m_sum.begin(); it != m_sum.end(); ++it)
		{
			if ((*it) != 0.)
				(*it_deriv) = 0.;
			else
				(*it_deriv) = INFINITY;

			++it_deriv;
		}
	}

	void Perceptron_logistic::activate()
	{
		sum();
		m_out = m_sum;

		vector<double>::iterator it_out = m_out.begin();
		for (vector<double>::iterator it = m_sum.begin(); it != m_sum.end(); ++it)
		{
			(*it_out) = 1. / (1. + exp(-(*it)));

			++it_out;
		}
	}

	void Perceptron_logistic::derivate()
	{
		m_derivate = m_sum;

		vector<double>::iterator it_deriv = m_derivate.begin();
		for (vector<double>::iterator it = m_out.begin(); it != m_out.end(); ++it)
		{
			(*it_deriv) = (*it)*(1. - (*it));

			++it_deriv;
		}
	}

	void Perceptron_tanH::activate()
	{
		sum();
		m_out = m_sum;

		vector<double>::iterator it_out = m_out.begin();
		for (vector<double>::iterator it = m_sum.begin(); it != m_sum.end(); ++it)
		{
			(*it_out) = (2. / (1. + exp(-2 * (*it)))) - 1.;

			++it_out;
		}
	}

	void Perceptron_tanH::derivate()
	{
		m_derivate = m_sum;

		vector<double>::iterator it_deriv = m_derivate.begin();
		for (vector<double>::iterator it = m_out.begin(); it != m_out.end(); ++it)
		{
			(*it_deriv) = 1. - (*it)*(*it);

			++it_deriv;
		}
	}

	void Perceptron_arcTan::activate()
	{
		sum();
		m_out = m_sum;

		vector<double>::iterator it_out = m_out.begin();
		for (vector<double>::iterator it = m_sum.begin(); it != m_sum.end(); ++it)
		{
			(*it_out) = atan(*it);

			++it_out;
		}
	}

	void Perceptron_arcTan::derivate()
	{
		m_derivate = m_sum;

		vector<double>::iterator it_deriv = m_derivate.begin();
		for (vector<double>::iterator it = m_sum.begin(); it != m_sum.end(); ++it)
		{
			(*it_deriv) = 1. / ((*it)*(*it) + 1.);

			++it_deriv;
		}
	}

	void Perceptron_softsign::activate()
	{
		sum();
		m_out = m_sum;

		vector<double>::iterator it_out = m_out.begin();
		for (vector<double>::iterator it = m_sum.begin(); it != m_sum.end(); ++it)
		{
			(*it_out) = (*it) / (1. + abs(*it));

			++it_out;
		}
	}

	void Perceptron_softsign::derivate()
	{
		m_derivate = m_sum;

		vector<double>::iterator it_deriv = m_derivate.begin();
		for (vector<double>::iterator it = m_sum.begin(); it != m_sum.end(); ++it)
		{
			(*it_deriv) = 1. / ((abs((*it)) + 1.)*(abs((*it)) + 1.));

			++it_deriv;
		}
	}

	void Perceptron_rectifier::activate()
	{
		sum();
		m_out = m_sum;

		vector<double>::iterator it_out = m_out.begin();
		for (vector<double>::iterator it = m_sum.begin(); it != m_sum.end(); ++it)
		{
			if ((*it) >= 0.)
				(*it_out) = (*it);
			else
				(*it_out) = 0.;

			++it_out;
		}
	}

	void Perceptron_rectifier::derivate()
	{
		m_derivate = m_sum;

		vector<double>::iterator it_deriv = m_derivate.begin();
		for (vector<double>::iterator it = m_sum.begin(); it != m_sum.end(); ++it)
		{
			if ((*it) < 0.)
				(*it_deriv) = 0.;
			else
				(*it_deriv) = 1.;

			++it_deriv;
		}
	}

	void Perceptron_rectifier_param::activate()
	{
		sum();
		m_out = m_sum;

		vector<double>::iterator it_out = m_out.begin();
		for (vector<double>::iterator it = m_sum.begin(); it != m_sum.end(); ++it)
		{
			if ((*it) >= 0.)
				(*it_out) = (*it);
			else
				(*it_out) = m_a*(*it);

			++it_out;
		}
	}

	void Perceptron_rectifier_param::derivate()
	{
		m_derivate = m_sum;

		vector<double>::iterator it_deriv = m_derivate.begin();
		for (vector<double>::iterator it = m_sum.begin(); it != m_sum.end(); ++it)
		{
			if ((*it) < 0.)
				(*it_deriv) = m_a;
			else
				(*it_deriv) = 1.;

			++it_deriv;
		}
	}

	void Perceptron_ELU::activate()
	{
		sum();
		m_out = m_sum;

		vector<double>::iterator it_out = m_out.begin();
		for (vector<double>::iterator it = m_sum.begin(); it != m_sum.end(); ++it)
		{
			if ((*it) >= 0.)
				(*it_out) = (*it);
			else
				(*it_out) = m_a*(exp(*it) - 1.);

			++it_out;
		}
	}

	void Perceptron_ELU::derivate()
	{
		m_derivate = m_sum;

		vector<double>::iterator it_deriv = m_derivate.begin();
		vector<double>::iterator it_out = m_out.begin();
		for (vector<double>::iterator it = m_sum.begin(); it != m_sum.end(); ++it)
		{
			if ((*it) < 0.)
				(*it_deriv) = m_a + (*it_out);
			else
				(*it_deriv) = 1.;

			++it_deriv;
			++it_out;
		}
	}

	void Perceptron_softPlus::activate()
	{
		sum();
		m_out = m_sum;

		vector<double>::iterator it_out = m_out.begin();
		for (vector<double>::iterator it = m_sum.begin(); it != m_sum.end(); ++it)
		{
			(*it_out) = log(1. + exp(*it));

			++it_out;
		}
	}

	void Perceptron_softPlus::derivate()
	{
		m_derivate = m_sum;

		vector<double>::iterator it_deriv = m_derivate.begin();
		for (vector<double>::iterator it = m_sum.begin(); it != m_sum.end(); ++it)
		{
			(*it_deriv) = 1. / (1. + exp(-(*it)));

			++it_deriv;
		}
	}

	void Perceptron_bent_identity::activate()
	{
		sum();
		m_out = m_sum;

		vector<double>::iterator it_out = m_out.begin();
		for (vector<double>::iterator it = m_sum.begin(); it != m_sum.end(); ++it)
		{
			(*it_out) = (sqrt((*it)*(*it) + 1.) - 1.) / 2. + (*it);

			++it_out;
		}
	}

	void Perceptron_bent_identity::derivate()
	{
		m_derivate = m_sum;

		vector<double>::iterator it_deriv = m_derivate.begin();
		for (vector<double>::iterator it = m_sum.begin(); it != m_sum.end(); ++it)
		{
			(*it_deriv) = (*it) / (2. * sqrt((*it)*(*it) + 1.)) + 1.;

			++it_deriv;
		}
	}

	void Perceptron_sinusoid::activate()
	{
		sum();
		m_out = m_sum;

		vector<double>::iterator it_out = m_out.begin();
		for (vector<double>::iterator it = m_sum.begin(); it != m_sum.end(); ++it)
		{
			(*it_out) = sin(*it);

			++it_out;
		}
	}

	void Perceptron_sinusoid::derivate()
	{
		m_derivate = m_sum;

		vector<double>::iterator it_deriv = m_derivate.begin();
		for (vector<double>::iterator it = m_sum.begin(); it != m_sum.end(); ++it)
		{
			(*it_deriv) = cos(*it);

			++it_deriv;
		}
	}

	void Perceptron_sinc::activate()
	{
		sum();
		m_out = m_sum;

		vector<double>::iterator it_out = m_out.begin();
		for (vector<double>::iterator it = m_sum.begin(); it != m_sum.end(); ++it)
		{
			if ((*it) == 0.)
				(*it_out) = 1.;
			else
				(*it_out) = sin(*it) / (*it);

			++it_out;
		}
	}

	void Perceptron_sinc::derivate()
	{
		m_derivate = m_sum;

		vector<double>::iterator it_deriv = m_derivate.begin();
		for (vector<double>::iterator it = m_sum.begin(); it != m_sum.end(); ++it)
		{
			if ((*it) == 0.)
				(*it_deriv) = 0.;
			else
				(*it_deriv) = cos((*it)) / (*it) + sin((*it)) / ((*it)*(*it));

			++it_deriv;
		}
	}

	void Perceptron_gaussian::activate()
	{
		sum();
		m_out = m_sum;

		vector<double>::iterator it_out = m_out.begin();
		for (vector<double>::iterator it = m_sum.begin(); it != m_sum.end(); ++it)
		{
			(*it_out) = exp(-(*it)*(*it));

			++it_out;
		}
	}

	void Perceptron_gaussian::derivate()
	{
		m_derivate = m_sum;

		vector<double>::iterator it_deriv = m_derivate.begin();
		for (vector<double>::iterator it = m_sum.begin(); it != m_sum.end(); ++it)
		{
			(*it_deriv) = -2. * exp(-(*it)*(*it));

			++it_deriv;
		}
	}

} // namespace SNN_network
