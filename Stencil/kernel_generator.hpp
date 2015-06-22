//#include <CL/cl.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <string>
#include <fstream>
//#include <ctime>
//#include "CLUtil.hpp"
#include <algorithm>
//#include "error_handling_methods.cpp"
//#include "CLPerfMarker.h"

using namespace std;

class KernelGenerator
{
	public:
		//KernelGenerator();

		void testfunktion();

		string generateKernelString(string name , int *positions, float *weights, int numberPoints, int edgewith);
};