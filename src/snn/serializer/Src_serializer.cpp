//============================================================================
// Name        : Src_serializer.cpp
// Authors     : Guillaume Sarthou
// EMail       : open.pode@gmail.com
// Date		   : 23 jul. 2017
// Version     : V2.1
// Copyright   : This file is part of SNN_network project which is released under
//               MIT license.
//============================================================================

#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include "snn/serializer/Src_serializer.h"

namespace SNN
{

	void Src_serializer::save(string file_name, Network& net)
	{
		bool h_file = false;
		if (file_name.find(".h") != string::npos)
			h_file = true;

		size_t pose = file_name.find(".");
		string var_name = file_name.substr(0, pose);

		string h_define = var_name;
		std::transform(h_define.begin(), h_define.end(), h_define.begin(), ::toupper);
		h_define += "_H";

		ofstream outfile;
		outfile.open(file_name, ios::binary | ios::out);

		if (h_file)
		{
			string str = "#ifndef " + h_define + "\n#define " + h_define + "\n\n";
			outfile.write(str.c_str(), str.length());
		}

		string inc = "#include <vector>\n\n";
		outfile.write(inc.c_str(), inc.length());

		string tab = "static const std::vector<char> " + var_name + " = {";
		outfile.write(tab.c_str(), tab.length());

		vector<char> encode_net = encode(net);

		char hex[16] = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A',   'B','C','D','E','F' };

		std::string str;
		for (int i = 0; i < encode_net.size(); ++i)
		{
			const char ch = encode_net[i];
			str += "0x";
			str.append(&hex[(ch & 0xF0) >> 4], 1);
			str.append(&hex[ch & 0xF], 1);
			if(i < encode_net.size() - 1)
				str += ",";
		}
		str += "};";

		if (h_file)
			str += "\n\n#endif";

		outfile.write(str.c_str(), str.length());
		outfile.close();
	}

	void Src_serializer::load(vector<char>& encode_net, Network& net)
	{
		net = decode(encode_net);
	}
}