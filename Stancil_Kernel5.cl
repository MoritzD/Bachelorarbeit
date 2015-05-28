/*
*	latest approach of a even more optimized version of #4 using local memory
*	performance has to be tested!
*/

__kernel void Stancil4_1(__global float* in, __global float* out, 
					int width, int height,
					 __local float* one, __local float* two, 
					 __local float* three, __local float* prefetchSpace)//, __local float* Buffer)
{
	int localWidth = get_local_size(0)+2;

	/*__local float one[localWidth];
	__local float two[localWidth];
	__local float three[localWidth];
	__local float prefetchSpace[localWidth];
*/


	int globalIDx = get_global_id(0);
	int localIDx = get_local_id(0)+1;
	//int localIDy = get_local_id(1);
	int group = get_group_id(0);
	int pos = globalIDx + 1 + width;
	//int localPos = localIDx + localWidth;
	event_t event;
	int helper;

	/*one[localIDx] = in[pos - width];
	two[localIDx] = in[pos];
	three[localIDx] = in[pos + width];
	if(localIDx == 0 ){
		two[localPos - 1] = in[pos - 1];
	}
	else if(localIDx == (localWidth-3)){
		two[localPos + 1] = in[pos + 1];
	}
	else if(localIDx == 1){
		three[localPos - 2 + localWidth] = in[pos - 2 + width];
	}
	else if(localIDx == (localWidth-4)){
		three[localPos + 2 + localWidth] = in[pos + 2 + width];
	}

	barrier(CLK_LOCAL_MEM_FENCE);
 */

	async_work_group_copy( one , in + group*(localWidth-2),localWidth, event);
	async_work_group_copy( two , in + width + group*(localWidth-2),localWidth, event);
	async_work_group_copy( three , in + width*2 + group*(localWidth-2),localWidth, event);

	//int line = from;
	for(int line = 1; line < height-1; line++){
		async_work_group_copy( prefetchSpace , in + (width*(line+2)) + group*(localWidth-2) ,localWidth, event);
	
		pos = globalIDx + 1 + (width*line);
		//localPos = localIDx + (localWidth * line);

		out[pos] = (two[localIDx-1]+two[localIDx+1]+one[localIDx]+three[localIDx])/4;	//-4*in[pos]+in[pos-1]+in[pos+1]+in[pos-width]+in[pos+width];
		
		helper = one;
		one = two;
		two = three;
		three = prefetchSpace;
		prefetchSpace = helper;
	}
}