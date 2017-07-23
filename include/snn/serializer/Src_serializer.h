//============================================================================
// Name        : Src_serializer.h
// Authors     : Guillaume Sarthou
// EMail       : open.pode@gmail.com
// Date		   : 23 jul. 2017
// Version     : V2.1
// Copyright   : This file is part of SNN_network project which is released under
//               MIT license.
//============================================================================

#ifndef SRC_SERIALIZER_H
#define SRC_SERIALIZER_H

#include <string>
#include <vector>
#include "snn/serializer/Serializer.h"

namespace SNN
{
	using namespace std;

	class Src_serializer : public Serializer
	{
	public:
		Src_serializer() {};
		virtual ~Src_serializer() {};

		void save(string file_name, Network& net);
		void load(vector<char>& encode_net, Network& net);
	private:
	};
}

#endif
