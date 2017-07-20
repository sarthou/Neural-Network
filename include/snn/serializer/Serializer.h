//============================================================================
// Name        : Serialize.h
// Authors     : Guillaume Sarthou
// EMail       : open.pode@gmail.com
// Date		   : 19 jul. 2017
// Version     : V2.1
// Copyright   : This file is part of SNN_network project which is released under
//               MIT license.
//============================================================================
#ifndef SERIALIZER_H
#define SERIALIZER_H

#include <string>
#include "snn/serializer/Serial_vector.h"

#include "snn/network/Network.h"

namespace SNN
{
	using namespace std;

	class Serializer
	{
	public:
		Serializer() {};
		~Serializer() {};

	//protected:
		vector<char> encode(Network& net);
		Network decode(vector<char>& data);
		
	};
}

#endif