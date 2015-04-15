/*
* Primary kernel to execute calculations.
*/
__kernel void stancel(__global float* in, __global float* out, 
					int width, int height)
{
	int num = get_global_id(0);
	//for (int doit = 0; doit < 10000000; doit++){

	if(num < width ||(num % width) == 0  || (num % width) == width-1 || num >= (width*height-width)){
	out[num] = in [num];
	}
	else{
		out[num] = -4*in[num]+in[num-1]+in[num+1]+in[num-width]+in[num+width]; 
	}
//	for(int ding = 0; ding< 1000 ;ding++){   //10000 
//		out[num] = out[num] * (out[num] + width);
//	}
}

__kernel void stancel2(__global float* in, __global float* out, 
					int width, int height)
{
	int globalID = get_global_id(0);
	int localID = get_local_id(0);
	int dim = get_work_dim();
	int group = get_group_id(0);
	int pos = (localID + 1 ) + ((group + 1) * width) ;//(num2 + 1) + ((num + 1) * width); 
	
	out[pos] = -4*in[pos]+in[pos-1]+in[pos+1]+in[pos-width]+in[pos+width]; 	
}

__kernel void copykernel(__global float* in, __global float* out, 
					int width, int height)
{
	int globalID = get_global_id(0);
	int localID = get_local_id(0);
	int dim = get_work_dim();
	int group = get_group_id(0);
	int pos = (localID + 1 ) + ((group + 1) * width) ;//(num2 + 1) + ((num + 1) * width); 
	
	out[pos] = in[pos]; 	
}