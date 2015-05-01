__kernel void stancel3(__global float* in, __global float* out, 
					int width, int height, __local float* Buffer)
{
	int2 globalID = (int2) (get_global_id(0), get_global_id(1));
	int2 localID = (int2) (get_local_id(0),get_local_id(1));
	int2 localSize = (int2) (get_local_size(0),get_local_size(1));
	int2 globalSize = (int2) (get_global_size(0),get_global_size(1));

	int2 group = (int2) (get_group_id(0),get_group_id(1));


	int dim = get_work_dim();
	int numberOfGroupsx = (globalSize.x/localSize.x);
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