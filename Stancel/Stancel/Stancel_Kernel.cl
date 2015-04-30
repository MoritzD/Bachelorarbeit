/*
* Primary kernel to execute calculations.
*/
__kernel void stancel1(__global float* in, __global float* out, 
					int width, int height)
{
	int num = get_global_id(0);
	//for (int doit = 0; doit < 10000000; doit++){

	if(num < width ||(num % width) == 0  || (num % width) == width-1 || num >= (width*height-width)){
	out[num] = in [num];
	}
	else{
		out[num] = (in[num-1]+in[num+1]+in[num-width]+in[num+width])/4;								//-4*in[num]+in[num-1]+in[num+1]+in[num-width]+in[num+width]; 
	}
//	for(int ding = 0; ding< 1000 ;ding++){   //10000 
//		out[num] = out[num] * (out[num] + width);
//	}
}

__kernel void stancel3(__global float* in, __global float* out, 
					int width, int height, __local float* Buffer)
{

	//int globalx = get_global_id(0);
	//int globaly = get_global_id(1);
	/*int localx = get_local_id(0);
	int localy = get_local_id(1);
	int localSizex = get_local_size(0);
	int localSizey = get_local_size(1);
	int globalSizex = get_global_size(0);
	int globalSizey = get_global_size(1);
*/
	int2 globalID = (int2) (get_global_id(0), get_global_id(1));
	int2 localID = (int2) (get_local_id(0),get_local_id(1));
	int2 localSize = (int2) (get_local_size(0),get_local_size(1));
	int2 globalSize = (int2) (get_global_size(0),get_global_size(1));

/*	int groupx = get_group_id(0);
	int groupy = get_group_id(1);
*/
	
	int2 group = (int2) (get_group_id(0),get_group_id(1));



	int dim = get_work_dim();
	int numberOfGroupsx = (globalSize.x/localSize.x);
	int pos = (globalID.x+1)+(globalID.y+1)*width;//(localID + 1 ) + ((group + 1) * width) ;//(num2 + 1) + ((num + 1) * width); 
	int localPos = (localID.x+1) + (localID.y+1)*(localSize.x+2);
	int loadIndex = localID.x + (localID.y * localSize.x);
	int globalStartPos = group.x * localSize.x + (group.y*localSize.y)* (globalSize.x+2); 
	//int globalEndPos = globalStartPos + localSizex+1 + (localSizey+1) * (globalSizex+2);
	int numcopys = (localSize.x+2) * (localSize.y+2);
	int globalLoadIndex = 0; // globalStartPos + loadIndex + (loadIndex/(localSizex+2))*(globalSizex-localSizex);

	//local float Buffer[(localSizey+2)*(localSizex+2)];
	
/*	
	using: 
	async_work_group_copy (	Buffer , in ,width*height, event);
	instead is a VERY BAAAAAD IDEA!!!!!!
	Time             SPS 
	71.7796          226485 per iterration!! 
*/	

	while(loadIndex < numcopys){
		globalLoadIndex = globalStartPos + loadIndex + (loadIndex/(localSize.x+2))*(globalSize.x-localSize.x);
		Buffer[loadIndex] = in[globalLoadIndex];
		loadIndex = loadIndex + (localSize.x*localSize.y);
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	out[pos] = (Buffer[localPos-1] + Buffer[localPos+1] + Buffer[localPos-(localSize.x+2)] + Buffer[localPos+(localSize.x+2)])/4;           //-4*Buffer[localPos] + Buffer[localPos-1] + Buffer[localPos+1] + Buffer[localPos-(localSizex+2)] + Buffer[localPos+(localSizex+2)]; //loadIndex;//(localSizex+2); //(globalSize/localSizex); //group + group2 * (globalSize/localSizex); //-4*in[pos]+in[pos-1]+in[pos+1]+in[pos-width]+in[pos+width]; 	

}

__kernel void stancel2(__global float* in, __global float* out, 
					int width, int height)
{
	int globalID = get_global_id(0);
	int localID = get_local_id(0);
	int dim = get_work_dim();
	int group = get_group_id(0);
	int pos = (localID + 1 ) + ((group + 1) * width) ;//(num2 + 1) + ((num + 1) * width); 
	
	out[pos] = (in[pos-1]+in[pos+1]+in[pos-width]+in[pos+width])/4;	//-4*in[pos]+in[pos-1]+in[pos+1]+in[pos-width]+in[pos+width];
}

__kernel void stancel4(__global float* in, __global float* out, 
					int width, int height) //, __local float* Buffer)
{
	int globalIDx = get_global_id(0);
	int localIDx = get_local_id(0);
	int localIDy = get_local_id(1);
	int group = get_group_id(0);
	int pos = 0; //globalID + 1 + width;
	//int from = (((height-2)*localIDy)/4)+1;
	//int to = (((height-2)*(localIDy+1))/4);

	//int line = from;
	for(int line = 1; line < height-1; line++){
		pos = globalIDx + 1 + (width*line);
		out[pos] = (in[pos-1]+in[pos+1]+in[pos-width]+in[pos+width])/4;	//-4*in[pos]+in[pos-1]+in[pos+1]+in[pos-width]+in[pos+width];
		
	}
}






__kernel void stancel4_1(__global float* in, __global float* out, 
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
