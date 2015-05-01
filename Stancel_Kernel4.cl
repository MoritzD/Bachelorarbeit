/*
*	latest approach of a Optimized code, but does not seems to be way better than #3
*/
__kernel void stancel4(__global float* in, __global float* out, 
					int width, int height) //, __local float* Buffer)
{
	int globalIDx = get_global_id(0);
	int localIDx = get_local_id(0);
	int localIDy = get_local_id(1);
	int group = get_group_id(0);
	int pos = 0; //globalID + 1 + width;
	int from = (((height-2)*localIDy)/4)+1;
	int to = (((height-2)*(localIDy+1))/4);

	//int line = from; 		< height-1
	for(int line = from; line <= to; line++){
		pos = globalIDx + 1 + (width*line);
		out[pos] = (in[pos-1]+in[pos+1]+in[pos-width]+in[pos+width])/4;	//-4*in[pos]+in[pos-1]+in[pos+1]+in[pos-width]+in[pos+width];
		
	}
}
