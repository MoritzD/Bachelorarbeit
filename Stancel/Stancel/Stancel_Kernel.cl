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

__kernel void stancel3(__global float* in, __global float* out, 
					int width, int height)
{
	int globalx = get_global_id(0);
	int globaly = get_global_id(1);
	int localx = get_local_id(0);
	int localy = get_local_id(1);
	int localSizex = get_local_size(0);
	int localSizey = get_local_size(1);
	int globalSizex = get_global_size(0);
	int globalSizey = get_global_size(1);

	int dim = get_work_dim();
	int groupx = get_group_id(0);
	int groupy = get_group_id(1);
	int numberOfGroupsx = (globalSizex/localSizex);
	int pos = (globalx)+(globaly)*width;//(localID + 1 ) + ((group + 1) * width) ;//(num2 + 1) + ((num + 1) * width); 
	int loadindex = localx + (localy * localSizex);
	int startPos = groupx * localSizex + (groupy*localSizey)* (globalSizex+2); 
	int endPos = startPos + localSizex+1 + (localSizey+1) * (globalSizex+2);

	//int count = 0;
	//while(count < 2){
	//	count = count +1;
	//}
	//local float Buffer[(localSizey+2)*(localSizex+2)];

	out[pos] = pos; //(globalSize/localSizex); //group + group2 * (globalSize/localSizex); //-4*in[pos]+in[pos-1]+in[pos+1]+in[pos-width]+in[pos+width]; 	
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