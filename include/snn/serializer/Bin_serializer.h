//============================================================================
// Name        : Bin_serializer.h
// Authors     : Guillaume Sarthou
// EMail       : open.pode@gmail.com
// Date		   : 22 jul. 2017
// Version     : V2.1
// Copyright   : This file is part of SNN_network project which is released under
//               MIT license.
//============================================================================

#ifndef BIN_SERIALIZER_H
#define BIN_SERIALIZER_H

#include <string>
#include "snn/serializer/Serializer.h"

namespace SNN
{
	using namespace std;

	class Bin_serializer : public Serializer
	{
	public:
		Bin_serializer() {};
		virtual ~Bin_serializer() {};

		void save(string file_name, Network& net);
		void load(string file_name, Network& net);
	private:
	};
}

#endif