/*
* Primary kernel to execute calculations. (not very efficient exept ATI FirePro W8000)
*/
__kernel void Stancil1(__global float* in, __global float* out, 
					int width, int height)
{
	int num = get_global_id(0);

	if(num < width ||(num % width) == 0  || (num % width) == width-1 || num >= (width*height-width)){
	out[num] = in [num];
	}
	else{
		out[num] = (in[num-1]+in[num+1]+in[num-width]+in[num+width])/4;								//-4*in[num]+in[num-1]+in[num+1]+in[num-width]+in[num+width]; 
	}
}
/*
*	Optimized kernel vor better performance on most devices! (exept ATI FirePro W8000)
*/
__kernel void Stancil3(__global float* in, __global float* out, 
					int width, int height, __local float* Buffer)
{
	int2 globalID = (int2) (get_global_id(0), get_global_id(1));
	int2 localID = (int2) (get_local_id(0),get_local_id(1));
	int2 localSize = (int2) (get_local_size(0),get_local_size(1));
	int2 globalSize = (int2) (get_global_size(0),get_global_size(1));

	int2 group = (int2) (get_group_id(0),get_group_id(1));


	int dim = get_work_dim();
	//int numberOfGroupsx = (globalSize.x/localSize.x);
	int pos = (globalID.x+1)+(globalID.y+1)*width;//(localID + 1 ) + ((group + 1) * width) ;//(num2 + 1) + ((num + 1) * width); 
	int localPos = (localID.x+1) + (localID.y+1)*(localSize.x+2);
	int loadIndex = localID.x + (localID.y * localSize.x);
	int globalStartPos = group.x * localSize.x + (group.y*localSize.y)* (globalSize.x+2); 
	int numcopys = (localSize.x+2) * (localSize.y+2);
	int globalLoadIndex = 0; // globalStartPos + loadIndex + (loadIndex/(localSizex+2))*(globalSizex-localSizex);
	
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
/*
*	First approach of a more optimized kernel but it's not working to good (and jutst to matrix sizes of 258)
*/
__kernel void Stancil2(__global float* in, __global float* out, 
					int width, int height)
{
	int2 globalID = (int2) (get_global_id(0), get_global_id(1));
	//int localID = get_local_id(0);
	//int dim = get_work_dim();
	//int group = get_group_id(0);
	int pos = globalID.x + 1 + (globalID.y+1)* width;//(num2 + 1) + ((num + 1) * width); 	
	out[pos] = (in[pos-1]+in[pos+1]+in[pos-width]+in[pos+width])/4;	//-4*in[pos]+in[pos-1]+in[pos+1]+in[pos-width]+in[pos+width];
}

/*
*	latest approach of a Optimized code, but does not seems to be way better than #3
*/
__kernel void Stancil4(__global float* in, __global float* out, 
					int width, int height) //, __local float* Buffer)
{
	int globalIDx = get_global_id(0);
	int localIDx = get_local_id(0);
	int localIDy = get_local_id(1);
	int group = get_group_id(0);
	int pos = 0; //globalID + 1 + width;
	int from = (((height-2)*localIDy)/4)+1;
	//int to = (((height-2)*(localIDy+1))/4);

	//int line = from; 		< height-1		<= to
	for(int line = from; line < height-1; line++){
		pos = globalIDx + 1 + (width*line);
		out[pos] = (in[pos-1]+in[pos+1]+in[pos-width]+in[pos+width])/4;	//-4*in[pos]+in[pos-1]+in[pos+1]+in[pos-width]+in[pos+width];
		
	}
}




/*
*	latest approach of a even more optimized version of #4 using local memory
*	performance has to be tested!
*
*	Note: Not working in the moment; i have no idea whats the problem... spent several hours trying to fix it... still no idea
* Jens version
*/

__kernel void Stancil4_1(__global float* in, __global float* out, 
					int width, int height,
					 __local float* one, __local float* two, 
					 __local float* three, __local float* prefetchSpace, __local float* helper)//, __local float* Buffer)
{
	int localWidth = get_local_size(0)+2;
	int globalIDx = get_global_id(0);
	int localIDx = get_local_id(0)+1;
	int group = get_group_id(0);
	int pos = globalIDx + 1 + width;

	__local float* temp = 0;


	event_t event = async_work_group_copy( one , in + (group*(localWidth-2) ) ,localWidth, 0);
	async_work_group_copy( two , in + width + group*(localWidth-2),localWidth, event);
	async_work_group_copy( three , in + width*2 + group*(localWidth-2),localWidth, event);

	for(int line = 1; line < 3; line++){
		barrier(CLK_LOCAL_MEM_FENCE);
		wait_group_events (&event, 1);
		event = async_work_group_copy( prefetchSpace , in + (width*(line+2)) + group*(localWidth-2) ,localWidth, 0);

		pos = globalIDx + 1 + (width*line);


		out[pos] = (two[localIDx-1]+two[localIDx+1]+one[localIDx]+three[localIDx])/4;	//-4*in[pos]+in[pos-1]+in[pos+1]+in[pos-width]+in[pos+width];

		temp = one;
		one = two;
		two = three;
		three = prefetchSpace;
		prefetchSpace = temp;
	}
}


__kernel void dynamicStancil1(__global float* in, __global float* out, 
					int width, int height, __global int* positions, __global float* weights,
					int numberPoints, int edgewith){
/*
*	__global int* positions, __global float* weights, int numberPoints)
*	Problem with the way the positions are handeld: when is a possition allowet to be in anoter line and when not?
*	idea vor a better way: set up a Struct or whatever to goup two int values together to one coordinate ->
*	its clear now wether a possition is ment to be in a diffrent line ore not (overflow)
*	new Way the position list is handeld: {x1,y1, x2,y2, x3,y3, x4,y4} wehre x and y are always the difference from the 
*	output point. e.g.: "normal" Stancil would be: {0,-1, -1,0, 1,0, 0,1}
*/


/*
Psoydo code:

	do for all points:

	sum = 0;
	for(int i = 0; i < numberPoints; i++){
			sum += (pos+posititions[i]) * weights[i] 
		if (pos+positition[i]) outOfBounce
			abbord
			
	}
	out[pos] = sum/numberPoints;
*/	
	int pos = get_global_id(0) + edgewith + (get_global_id(1)+edgewith)*width;

	float sum = 0;
	int lookAt = 0;

	for(int i = 0; i < numberPoints*2; i = i + 2){
		lookAt = pos + positions[i] + positions[i+1]*width;
		sum += in[lookAt] * weights[i/2];
	}
	out[pos] = sum/numberPoints;
}

/*
*	Not doing any aktual work; just vor testing.
*/

__kernel void dynamicStancil1_Copy_kernel(__global float* in, __global float* out, 
					int width, int height, __global int* positions, __global float* weights, int numberPoints){

	int2 globalID = (int2) (get_global_id(0), get_global_id(1));

	int pos = globalID.x + 1 + (globalID.y+1)*width;
	out[pos] = in[pos];
}