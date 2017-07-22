//============================================================================
// Name        : Bin_serializer.cpp
// Authors     : Guillaume Sarthou
// EMail       : open.pode@gmail.com
// Date		   : 22 jul. 2017
// Version     : V2.1
// Copyright   : This file is part of SNN_network project which is released under
//               MIT license.
//============================================================================

#include <vector>
#include <fstream>
#include <iostream>
#include "snn/serializer/Bin_serializer.h"

namespace SNN
{

	void Bin_serializer::save(string file_name, Network& net)
	{
		ofstream outfile;
		outfile.open(file_name, ios::binary | ios::out);
		vector<char> encode_net = encode(net);
		string str(encode_net.begin(), encode_net.end());
		outfile.write(str.c_str(), str.length());
		outfile.close();
	}

	void Bin_serializer::load(string file_name, Network& net)
	{
		ifstream infile;
		infile.open(file_name, ios::binary | ios::in);
		infile.seekg(0, ios::end);
		streamoff length = infile.tellg();
		infile.seekg(0, ios::beg);
		char* buffer = new char[(size_t)length];
		infile.read(buffer, length);
		infile.close();

		//string str(buffer);
		vector<char> encode_net;
		for (unsigned int i = 0; i < length; i++)
			encode_net.push_back(buffer[i]);
		//copy(str.c_str(), str.c_str() + str.length(), encode_net.begin());
		delete buffer;

		net = decode(encode_net);
	}
}