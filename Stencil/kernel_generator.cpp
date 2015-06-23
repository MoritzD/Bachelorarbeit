#include "kernel_generator.hpp"

using namespace std;

void KernelGenerator::testfunktion(){
	cout << "it Worked, I AM THE TESTFUNKTION!!!" << endl;
}

string KernelGenerator::generateKernelString(string name, int *positions, float *weights, int numberPoints, int edgewith){
	stringstream kernelString;
	kernelString << "__kernel void " << name;
	kernelString << "(__global float* in, __global float* out, int width, int height){\n\t";
	kernelString << "int pos = get_global_id(0) + " << edgewith << " + (get_global_id(1) + "
				 << edgewith << " ) * width;\n\t";
	kernelString << "out[pos] = (";

	float posOffset;
	for(int i = 0; i < (numberPoints-1)*2; i = i + 2){
		if(weights[i/2] == 0)
			continue;
		
		else if(weights[i/2] == 1){
			if(positions[i+1] == 0){
				if(positions[i] == 0){
					kernelString << "(in[pos] ) + ";
					continue;
				}
				if(positions[i] >= 0){
					kernelString << "(in[pos + " << positions[i] 
					<< "] ) + ";
					continue;
				}else{
					kernelString << "(in[pos " << positions[i] 
					<< "] ) + ";
					continue;
				}
			}
			if(positions[i] == 0){
				if(positions[i+1] == 1){
					kernelString << "(in[pos + width] ) + ";
					continue;
				}
				if(positions[i+1] == -1){
					kernelString << "(in[pos - width] ) + ";
					continue;
				}
				if(positions[i+1] >= 0){
					kernelString << "(in[pos + " << positions[i+1] << " * width] ) + ";
					continue;
				}else{
					kernelString << "(in[pos " << positions[i+1] << " * width] ) + ";
					continue;
				}

			}
			
			if(positions[i] >= 0){
				if(positions[i+1] == 1){
					kernelString << "(in[pos + " << positions[i] << " + width] ) + ";
					continue;
				}
				if(positions[i+1] == -1){
					kernelString << "(in[pos + " << positions[i] << " - width] ) + ";
					continue;
				}
				if(positions[i+1] >= 1){
					kernelString << "(in[pos + " << positions[i] << " + (width * " 
								<< positions[i+1] << "] ) + ";
					continue;  
				}
				else{
					kernelString << "(in[pos + " << positions[i] << " + (width * (" 
								<< positions[i+1] << ")] ) + ";
					continue;  
				}

			}
			else{
				if(positions[i+1] == 1){
					kernelString << "(in[pos " << positions[i] << " + width] ) + ";
					continue;
				}
				if(positions[i+1] == -1){
					kernelString << "(in[pos " << positions[i] << " - width] ) + ";
					continue;
				}
				if(positions[i+1] >= 1){
					kernelString << "(in[pos " << positions[i] << " + (width * " 
								<< positions[i+1] << "] ) + ";
					continue;  
				}
				else{
					kernelString << "(in[pos " << positions[i] << " + (width * (" 
								<< positions[i+1] << ")] ) + ";
					continue;  
				}
			}
		}
		else {
			if(positions[i+1] == 0){
				if(positions[i] == 0){
					kernelString << "(in[pos] * " << weights[i/2] << ") + ";
					continue;
				}
				if(positions[i] >= 0){
					kernelString << "(in[pos + " << positions[i] 
					<< "] * " << weights[i/2] << ") + ";
					continue;
				}else{
					kernelString << "(in[pos " << positions[i] 
					<< "] * " << weights[i/2] << ") + ";
					continue;
				}
			}
			if(positions[i] == 0){
				if(positions[i+1] == 1){
					kernelString << "(in[pos + width] * " << weights[i/2] << ") + ";
					continue;
				}
				if(positions[i+1] == -1){
					kernelString << "(in[pos - width] * " << weights[i/2] << ") + ";
					continue;
				}
				if(positions[i+1] >= 0){
					kernelString << "(in[pos + " << positions[i+1] << " * width] * " << weights[i/2] << ") + ";
					continue;
				}else{
					kernelString << "(in[pos " << positions[i+1] << " * width] * " << weights[i/2] << ") + ";
					continue;
				}

			}
			
			if(positions[i] >= 0){
				if(positions[i+1] == 1){
					kernelString << "(in[pos + " << positions[i] << " + width] * " << weights[i/2] << ") + ";
					continue;
				}
				if(positions[i+1] == -1){
					kernelString << "(in[pos + " << positions[i] << " - width] * " << weights[i/2] << ") + ";
					continue;
				}
				if(positions[i+1] >= 1){
					kernelString << "(in[pos + " << positions[i] << " + (width * " 
								<< positions[i+1] << "] * " << weights[i/2] << ") + ";
					continue;  
				}
				else{
					kernelString << "(in[pos + " << positions[i] << " + (width * (" 
								<< positions[i+1] << ")] * " << weights[i/2] << ") + ";
					continue;  
				}

			}
			else{
				if(positions[i+1] == 1){
					kernelString << "(in[pos " << positions[i] << " + width] * " << weights[i/2] << ") + ";
					continue;
				}
				if(positions[i+1] == -1){
					kernelString << "(in[pos " << positions[i] << " - width] * " << weights[i/2] << ") + ";
					continue;
				}
				if(positions[i+1] >= 1){
					kernelString << "(in[pos " << positions[i] << " + (width * " 
								<< positions[i+1] << "] * " << weights[i/2] << ") + ";
					continue;  
				}
				else{
					kernelString << "(in[pos " << positions[i] << " + (width * (" 
								<< positions[i+1] << ")] * " << weights[i/2] << ") + ";
					continue;  
				}
			}
		}
	}


	
	int i = (numberPoints-1)*2;

	if(weights[i/2] != 0){
			
		if(weights[i/2] == 1){
			if(positions[i+1] == 0){
				if(positions[i] == 0){
					kernelString << "(in[pos] )";
				}
				else if(positions[i] >= 0){
					kernelString << "(in[pos + " << positions[i] 
					<< "] )";
				}else{
					kernelString << "(in[pos " << positions[i] 
					<< "] )";
				}
			}
			else if(positions[i] == 0){
				if(positions[i+1] == 1){
					kernelString << "(in[pos + width] )";
				}
				else if(positions[i+1] == -1){
					kernelString << "(in[pos - width] )";
				}
				else if(positions[i+1] >= 0){
					kernelString << "(in[pos + " << positions[i+1] << " * width] )";
				}else{
					kernelString << "(in[pos " << positions[i+1] << " * width] )";
				}

			}
			
			else if(positions[i] >= 0){
				if(positions[i+1] == 1){
					kernelString << "(in[pos + " << positions[i] << " + width] )";
				}
				else if(positions[i+1] == -1){
					kernelString << "(in[pos + " << positions[i] << " - width] )";
				}
				else if(positions[i+1] >= 1){
					kernelString << "(in[pos + " << positions[i] << " + (width * " 
								<< positions[i+1] << "] )";  
				}
				else{
					kernelString << "(in[pos + " << positions[i] << " + (width * (" 
								<< positions[i+1] << ")] )";  
				}

			}
			else{
				if(positions[i+1] == 1){
					kernelString << "(in[pos " << positions[i] << " + width] )";
				}
				else if(positions[i+1] == -1){
					kernelString << "(in[pos " << positions[i] << " - width] )";
				}
				else if(positions[i+1] >= 1){
					kernelString << "(in[pos " << positions[i] << " + (width * " 
								<< positions[i+1] << "] )";  
				}
				else{
					kernelString << "(in[pos " << positions[i] << " + (width * (" 
								<< positions[i+1] << ")] )";  
				}
			}
		}
		else{
			if(positions[i+1] == 0){
				if(positions[i] == 0){
					kernelString << "(in[pos] * " << weights[i/2] << ")";
				}
				else if(positions[i] >= 0){
					kernelString << "(in[pos + " << positions[i] 
					<< "] * " << weights[i/2] << ")";
				}else{
					kernelString << "(in[pos " << positions[i] 
					<< "] * " << weights[i/2] << ")";
				}
			}
			if(positions[i] == 0){
				if(positions[i+1] == 1){
					kernelString << "(in[pos + width] * " << weights[i/2] << ")";
				}
				else if(positions[i+1] == -1){
					kernelString << "(in[pos - width] * " << weights[i/2] << ")";
				}
				else if(positions[i+1] >= 0){
					kernelString << "(in[pos + " << positions[i+1] << " * width] * " << weights[i/2] << ")";
				}else{
					kernelString << "(in[pos " << positions[i+1] << " * width] * " << weights[i/2] << ")";
				}

			}
			
			else if(positions[i] >= 0){
				if(positions[i+1] == 1){
					kernelString << "(in[pos + " << positions[i] << " + width] * " << weights[i/2] << ")";
				}
				else if(positions[i+1] == -1){
					kernelString << "(in[pos + " << positions[i] << " - width] * " << weights[i/2] << ")";
				}
				else if(positions[i+1] >= 1){
					kernelString << "(in[pos + " << positions[i] << " + (width * " 
								<< positions[i+1] << "] * " << weights[i/2] << ")";  
				}
				else{
					kernelString << "(in[pos + " << positions[i] << " + (width * (" 
								<< positions[i+1] << ")] * " << weights[i/2] << ")";  
				}

			}
			else{
				if(positions[i+1] == 1){
					kernelString << "(in[pos " << positions[i] << " + width] * " << weights[i/2] << ")";
				}
				else if(positions[i+1] == -1){
					kernelString << "(in[pos " << positions[i] << " - width] * " << weights[i/2] << ")";
				}
				else if(positions[i+1] >= 1){
					kernelString << "(in[pos " << positions[i] << " + (width * " 
								<< positions[i+1] << "] * " << weights[i/2] << ")";  
				}
				else{
					kernelString << "(in[pos " << positions[i] << " + (width * (" 
								<< positions[i+1] << ")] * " << weights[i/2] << ")";  
				}
			}
		}
	}

	int sumOfWeights = 0;
	for(int e = 0; e < numberPoints; e++){
		sumOfWeights += weights[e];
	}

	kernelString << ") / " << sumOfWeights << ";\n";

	
	kernelString << "}" ;
	return kernelString.str();
}


/*

__kernel void dynamicStencil1(__global float* in, __global float* out, 
					int width, int height, __global int* positions, __global float* weights,
					int numberPoints, int edgewith){


	int pos = get_global_id(0) + edgewith + (get_global_id(1)+edgewith)*width;

	float sum = 0;
	int lookAt = 0;

	for(int i = 0; i < numberPoints*2; i = i + 2){
		lookAt = pos + positions[i] + positions[i+1]*width;
		sum += in[lookAt] * weights[i/2];
	}
	out[pos] = sum/numberPoints;
}

__kernel void Stencil2(__global float* in, __global float* out, 
					int width, int height)
{
	int2 globalID = (int2) (get_global_id(0), get_global_id(1));
	
	int pos = globalID.x + 1 + (globalID.y+1)* width;//(num2 + 1) + ((num + 1) * width); 	
	out[pos] = (in[pos-1]+in[pos+1]+in[pos-width]+in[pos+width])/4;	//-4*in[pos]+in[pos-1]+in[pos+1]+in[pos-width]+in[pos+width];
}


*/