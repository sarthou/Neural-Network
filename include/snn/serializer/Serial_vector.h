//============================================================================
// Name        : Serial_vector.h
// Authors     : Guillaume Sarthou
// EMail       : open.pode@gmail.com
// Date		   : 19 jul. 2017
// Version     : V2.1
// Copyright   : This file is part of SNN_network project which is released under
//               MIT license.
//============================================================================
#ifndef SERIAL_VECTOR
#define SERIAL_VECTOR

#include <vector>
#include "snn/trainer/Trainer.h"
#include <cstdint>

namespace SNN
{
	using namespace std;

	class Serial_vector
	{
	public:
		Serial_vector() { m_index = 0; m_data.resize(0); };
		Serial_vector(vector<char>& data) { m_index = 0; m_data = data; };
		~Serial_vector() {};

		void to_begin() { m_index = 0; };
		void set_index(unsigned int index) { m_index = index; };

		void push_back(char data) { m_data.push_back(data); };
		void push_back(unsigned char data) { m_data.push_back((char)data); };
		void push_back(int data)
		{
			m_data.push_back((char)data);
			m_data.push_back((char)(data >> 8));
		};
		void push_back(unsigned int data)
		{
			m_data.push_back((char)data);
			m_data.push_back((char)(data >> 8));
		};
		void push_back(float data)
		{
			char result[sizeof(float)];
			memcpy(result, &data, sizeof(data));
			for(unsigned int i = 0; i < sizeof(data); i++)
				m_data.push_back(result[i]);
		};

		char get_next_char() { return m_data[m_index++]; };
		unsigned char get_next_uchar() { return (unsigned char)m_data[m_index++]; };
		int16_t get_next_int()
		{
			int16_t result;
			result = m_data[m_index++];
			result |= m_data[m_index++] << 8;
			return result;
		}
		uint16_t get_next_uint()
		{
			uint16_t result;
			result = m_data[m_index++];
			result |= m_data[m_index++] << 8;
			return result;
		}
		float get_next_float()
		{
			float result = 0;
			char data[sizeof(float)];
			
			for (unsigned int i = 0; i < sizeof(result); i++)
				data[i] = m_data[m_index++];

			memcpy(&result, data, sizeof(result));
			return result;
		}

		unsigned int size() { return m_data.size(); };

		vector<char> get_data() { return m_data; };

	private:
		vector<char> m_data;
		unsigned int m_index;
	};
}
#endif // !SERIAL_VECTOR

