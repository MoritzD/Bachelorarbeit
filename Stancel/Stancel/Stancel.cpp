// Stancel.cpp : Defines the entry point for the console application.
//

//#include "stdafx.h"

#include "Stancel.hpp"

  

 
using namespace std;

/* convert the kernel file into a string */
int convertToString(const char *filename, std::string& s)
{
	size_t size;
	char*  str;
	std::fstream f(filename, (std::fstream::in | std::fstream::binary));

	if (f.is_open())
	{
		size_t fileSize;
		f.seekg(0, std::fstream::end);
		size = fileSize = (size_t)f.tellg();
		f.seekg(0, std::fstream::beg);
		str = new char[size + 1];
		if (!str)
		{
			f.close();
			return 0;
		}

		f.read(str, fileSize);
		f.close();
		str[size] = '\0';
		s = str;
		delete[] str;
		return 0;
	}
	cout << "Error: failed to open file\n:" << filename << endl;
	return FAILURE;
}



int main(int argc, char* argv[])
{
	sampleTimer = new SDKTimer();
	timer = sampleTimer->createTimer();
	
	if(readArgs(argc, argv) == SDK_FAILURE){
		freeResources();
		return FAILURE;
	}


	if (getPlatforms() == FAILURE){ 
		freeResources();
		return FAILURE; 
	}

	if (getDevice() == END){ 
		freeResources();
		return SUCCESS; 
	}
	
	/*Step 3: Create context.*/
	cl_context context = clCreateContext(NULL, 1, aktiveDevice, NULL, NULL, NULL);


	/*Step 4: Creating command queue associate with the context.*/
	cl_command_queue commandQueue = clCreateCommandQueue(context, *aktiveDevice, 0, NULL);

	/*Step 5: Create program object */
	sampleTimer->resetTimer(timer);
	sampleTimer->startTimer(timer);

	const char *filename = "Stancel_Kernel.cl";

	string KernelSource;
	status = convertToString(filename, KernelSource);
	const char *source = KernelSource.c_str();
	size_t sourceSize[] = { strlen(source) };
	if (VERBOSE){ cout << source << endl; }
	cl_program program = clCreateProgramWithSource(context, 1, &source, sourceSize, NULL);

	if(buildProgram(&program) != SUCCESS) {
		freeResources();
		return FAILURE;
	}
	sampleTimer->stopTimer(timer);
	times.buildProgram = sampleTimer->readTimer(timer);

	srand (static_cast <unsigned> (time(0)));

	input = (cl_float*)malloc(sizeof(cl_float) * width * height);
	fill(input, input + (width*height), 1.0);
	for (int i = width; i < (width*height)-width; i++){
		if(i%width == 0 || i%width == (width-1)){
			continue;
		}
		/*if(i == (width/2+(height/2)*width)){
			input[i] = 100;
			continue;
		}*/
		input[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/10));
	}
	
	output = (cl_float*)malloc(sizeof(cl_float) * width * height);
	memset(output, 0, sizeof(cl_float) * width * height);

	sampleTimer->resetTimer(timer);
	sampleTimer->startTimer(timer);
	// Create buffer for matrix A
	cl_mem BufferMatrixA = clCreateBuffer(
		context,
		CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
		sizeof(cl_float) * width * height,
		input,
		&status);
	if (status != SUCCESS){ 
		fprintf(stderr, "clCreateBuffer failed. (BufferMatrixA)\n");
		freeResources();
		return FAILURE;
	}

	// Create buffer for matrix B
	cl_mem BufferMatrixB = clCreateBuffer(
		context,
		CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
		sizeof(cl_float) * width * height,
		input,
		&status);
	if (status != SUCCESS){
		fprintf(stderr, "clCreateBuffer failed. (BufferMatrixB) %i\n", status);
		freeResources();
		return FAILURE;
	}
	
	cl_kernel 
	kernel = NULL, 
	kernelBackwards = NULL;
	
	/*Step 8: Create kernel object */
	switch (kernelVersion){
		case 1:
			kernel = clCreateKernel(program, "stancel1", NULL);
			kernelBackwards = clCreateKernel(program, "stancel1", NULL);
		break;
		case 2:
			kernel = clCreateKernel(program, "stancel2", NULL);
			kernelBackwards = clCreateKernel(program, "stancel2", NULL);
		break;
		case 3:
			kernel = clCreateKernel(program, "stancel3", NULL);
			kernelBackwards = clCreateKernel(program, "stancel3", NULL);
		break;
		case 4:
			kernel = clCreateKernel(program, "stancel4_1", NULL);
			kernelBackwards = clCreateKernel(program, "stancel4_1", NULL); 
		break;
	}

	/*Step 9: Sets Kernel arguments.*/
	status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&BufferMatrixA);
	status = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&BufferMatrixB);
	status = clSetKernelArg(kernel, 2, sizeof(cl_int), (void *)&width);
	status = clSetKernelArg(kernel, 3, sizeof(cl_int), (void *)&height);
	if (status != SUCCESS){	
		fprintf(stderr, "setting kernel arguments failed. \n");	
		cout << status << endl;
		getKernelArgSetError(status);
		freeResources();
		return FAILURE;
	}


	status = clSetKernelArg(kernelBackwards, 0, sizeof(cl_mem), (void *)&BufferMatrixB);
	status = clSetKernelArg(kernelBackwards, 1, sizeof(cl_mem), (void *)&BufferMatrixA);
	status = clSetKernelArg(kernelBackwards, 2, sizeof(cl_int), (void *)&width);
	status = clSetKernelArg(kernelBackwards, 3, sizeof(cl_int), (void *)&height);
	if (status != SUCCESS){	
		fprintf(stderr, "setting kernelBackwards arguments failed. \n");
		cout << status << endl;
		getKernelArgSetError(status);
		freeResources();
		return FAILURE;
	}

	sampleTimer->stopTimer(timer);
	times.setKernelArgs = sampleTimer->readTimer(timer);

	cout << "kernel Arguments are set; starting kernel now!" << endl;


	/*Step 10: Running the kernel.*/
	/*	Approach (1 and) 2 
	*	cl_uint work_dim = 1;
	*	size_t global_work_size[1] = {(height - 2)*(width - 2)};			//{ width * height };
	*	size_t local_work_size [1] = {global_work_size[0]/(height-2)};		//{2};				//{width - 2}; 
	*	
	*	//global_work_size[2] = width -2;
	*/																// first approach: every single stancel by itself
																	// second approach: one line a a time; as one local work goup (making tourble with big matrices, as soon as it gets bigger than what can be managed as one work item)
																	// third approach: picking 16 by 16 blocks (or what ever the hadware can handle) and calculate these at the same time. thervor using a work_dim of 2 (global and local) another benefit: posebilety to load buffer from global to local mem to gain performance
	/* Approach 3 (I feel like this is it ;) )*/					// whats better? deviding Matrix in equaly sized blocks (with might not have optimal size) or bilding optimal sized blocks and check in rightmost block for out of bounce?

	//Get maximum work goup size
	KernelWorkGroupInfo kernelInfo;
	status = kernelInfo.setKernelWorkGroupInfo(kernel, *aktiveDevice);
    CHECK_ERROR(status,0, "setKernelWrkGroupInfo failed");
    cout << "Max kernel work gorup size: " << kernelInfo.kernelWorkGroupSize << endl;
	
	cl_uint work_dim;
	size_t *global_work_size = (size_t*) malloc(2*sizeof(size_t));
	size_t *local_work_size = (size_t*) malloc(2*sizeof(size_t));

	switch (kernelVersion){
		case 1:
			work_dim = 1;
			global_work_size[0] = width * height;
			local_work_size = NULL;
		break;

		case 2:
			work_dim = 1;
			global_work_size[0] = (height - 2)*(width - 2);
			local_work_size[0] = global_work_size[0]/(height-2);
		break;

		case 3:
			work_dim = 2;
			global_work_size[0] = (width - 2);
			global_work_size[1] = (height - 2);
			local_work_size[0] = local_work_size[1] = 4;
			
			for (int i = min(global_work_size[0], (size_t) 16); i > 0; i--)		//(size_t)(sqrt(kernelInfo.kernelWorkGroupSize)) in min
			{
					if(global_work_size[0]%i == 0){
					local_work_size[0] = local_work_size[1] = i;
					break; 
				}
			}
			cout << "Using blocks of size: " << local_work_size[0] <<" ; "<< local_work_size[1] << endl;

			/* Create local mem objects to cash blocks in */
			status = clSetKernelArg(kernel, 4, (local_work_size[0] + 2) * (local_work_size[1] + 2) * sizeof(cl_float), NULL);
		    CHECK_OPENCL_ERROR(status, "clSetKernelArg failed. (local memory)");

		    status = clSetKernelArg(kernelBackwards, 4, (local_work_size[0] + 2) * (local_work_size[1] + 2) * sizeof(cl_float), NULL);
		    CHECK_OPENCL_ERROR(status, "clSetKernelArg failed. (local memory)");
		/*
			if(kernelInfo.kernelWorkGroupSize >= 1024){ // use bloks of 32*32
				local_work_size[0] = local_work_size [1] = 32;
			}
			else if(kernelInfo.kernelWorkGroupSize >= 256){ //use bloks of 16*16
				local_work_size[0] = local_work_size [1] = 16;
			}
			else if(kernelInfo.kernelWorkGroupSize >= 64){ // use bloks of 8*8
				local_work_size[0] = local_work_size [1] = 8;
			}
			//else{											// default case (16 should be supported anyway)
			//	size_t local_work_size [2] = {4,4};
			//}
		*/


			cout <<" height  and    width     "<< height << " " << width << endl;

			cout <<" global work size:  we ; he   "<< global_work_size[0] <<" ; "<< global_work_size[1] << endl;

			cout <<" lokal work size:  we ; he   "<< local_work_size[0] <<" ; " << local_work_size[1] << endl;
		break;
		case 4:
			//work_dim = 2;
			work_dim = 1;
			global_work_size[0] = (width - 2);
			//global_work_size[1] = 4;
			local_work_size[0] = min((cl_uint)64,(width-2));
			//local_work_size[1] = 4;

			status = clSetKernelArg(kernel, 4, (local_work_size[0] + 2) * sizeof(cl_float), NULL);
		    CHECK_OPENCL_ERROR(status, "clSetKernelArg failed. (local memory1)");

			status = clSetKernelArg(kernel, 5, (local_work_size[0] + 2) * sizeof(cl_float), NULL);
		    CHECK_OPENCL_ERROR(status, "clSetKernelArg failed. (local memory2)");

		    status = clSetKernelArg(kernel, 6, (local_work_size[0] + 2) * sizeof(cl_float), NULL);
		    CHECK_OPENCL_ERROR(status, "clSetKernelArg failed. (local memory3)");

		    status = clSetKernelArg(kernel, 7, (local_work_size[0] + 2) * sizeof(cl_float), NULL);
		    CHECK_OPENCL_ERROR(status, "clSetKernelArg failed. (local memory3)");



		    status = clSetKernelArg(kernelBackwards, 4, (local_work_size[0] + 2) * sizeof(cl_float), NULL);
		    CHECK_OPENCL_ERROR(status, "clSetKernelArg failed. (local memory4)");

		    status = clSetKernelArg(kernelBackwards, 5, (local_work_size[0] + 2) * sizeof(cl_float), NULL);
		    CHECK_OPENCL_ERROR(status, "clSetKernelArg failed. (local memory5)");

		    status = clSetKernelArg(kernelBackwards, 6, (local_work_size[0] + 2) * sizeof(cl_float), NULL);
		    CHECK_OPENCL_ERROR(status, "clSetKernelArg failed. (local memory6)");

		    status = clSetKernelArg(kernelBackwards, 7, (local_work_size[0] + 2) * sizeof(cl_float), NULL);
		    CHECK_OPENCL_ERROR(status, "clSetKernelArg failed. (local memory6)");


 
			cout <<" height  and    width     "<< height << " " << width << endl;

			cout <<" global work size:  we ; he   "<< global_work_size[0] <<" ; "<< global_work_size[1] << endl;
 
			cout <<" lokal work size:  we ; he   "<< local_work_size[0] <<" ; " << local_work_size[1] << endl;
		break;
	}



	sampleTimer->resetTimer(timer);
	sampleTimer->startTimer(timer);
	int timer2 = sampleTimer->createTimer();

	double timer2value = 0;

		if(SINGLETIME){ 
			cout << "Time \t\t SPS " << endl;
			} 

	for (int e = 0; e < iterations; e++){
		if(SINGLETIME){
			sampleTimer->resetTimer(timer2);
			sampleTimer->startTimer(timer2);
		}

		status = clEnqueueNDRangeKernel(commandQueue, kernel, work_dim, NULL, global_work_size, local_work_size, 0, NULL, &ndrEvt);
		if (status != SUCCESS){
			fprintf(stderr, "executing kernel failed. \n %i vgl %i\n ",status , CL_INVALID_WORK_ITEM_SIZE   ); //CL_INVALID_EVENT_WAIT_LIST CL_MEM_OBJECT_ALLOCATION_FAILURE CL_MEM_OBJECT_ALLOCATION_FAILURE  CL_INVALID_WORK_DIMENSION
			getExecutionError(status);
			freeResources();
			return FAILURE;
		}
		status = clFlush(commandQueue);

		eventStatus = CL_QUEUED;
		while (eventStatus != CL_COMPLETE)
		{
			status = clGetEventInfo(
				ndrEvt,
				CL_EVENT_COMMAND_EXECUTION_STATUS,
				sizeof(cl_int),
				&eventStatus,
				NULL);
			if (status != SUCCESS){
			fprintf(stderr, "clGetEventInfo failed. %i\n", status);
			freeResources();
			return FAILURE;
			}
		}
		
		status = clEnqueueNDRangeKernel(commandQueue, kernelBackwards, work_dim, NULL, global_work_size, local_work_size, 0, NULL, &ndrEvt);
		if (status != SUCCESS) fprintf(stderr, "executing kernel simply just for the second try failed. \n");
		status = clFlush(commandQueue);

		eventStatus = CL_QUEUED;
		while (eventStatus != CL_COMPLETE)
		{
			status = clGetEventInfo(
				ndrEvt,
				CL_EVENT_COMMAND_EXECUTION_STATUS,
				sizeof(cl_int),
				&eventStatus,
				NULL);
			if (status != SUCCESS){
			fprintf(stderr, "clGetEventInfo failed. %i\n", status);
			freeResources();
			return FAILURE;
		}
		}

		if(SINGLETIME){ 
			sampleTimer->stopTimer(timer2);
			timer2value = sampleTimer->readTimer(timer2);
			cout << timer2value  << " \t " << ((width - 2)*(height - 2))/ timer2value << endl;
		}
	}

	sampleTimer->stopTimer(timer);
	times.kernelExecuting = sampleTimer->readTimer(timer);
	cout << "Total executing time: " << times.kernelExecuting << endl;
	cout << "so for every run thats: " << (sampleTimer->readTimer(timer) / iterations) << endl;

	sampleTimer->resetTimer(timer);
	sampleTimer->startTimer(timer);
	/*Step 11: Read the cout put back to host memory.*/
	status = clEnqueueReadBuffer(commandQueue, BufferMatrixA, CL_TRUE, 0, width * height * sizeof(cl_float), output, 0, NULL, NULL);
	//status = clEnqueueReadBuffer(commandQueue, BufferMatrixB, CL_TRUE, 0, width * height * sizeof(cl_float), output, 0, NULL, NULL);

	sampleTimer->stopTimer(timer);
	times.writeBack = sampleTimer->readTimer(timer);


	if (VERBOSE){
		cout << "Input:" << endl;
		for (int y = 0; y < height; y++){
			for (int x = 0; x < width; x++){
				cout << *(input + x + y*width)<<" ";
			}
			cout << endl;
		}
		cout << endl << "Output:" << endl;
		for (int y = 0; y < height; y++){
			for (int x = 0; x < width; x++){
				cout << *(output + x + y*width)<<" ";
			}
			cout << endl;
		}
	}

	cout << "done; releasinng kernel now" << endl;
	sampleTimer->resetTimer(timer);
	sampleTimer->startTimer(timer);

	status = clReleaseKernel(kernel);				//Release kernel.
	status = clReleaseProgram(program);				//Release the program object.

	status = clReleaseMemObject(BufferMatrixA);		//Release mem object.
	status = clReleaseMemObject(BufferMatrixB);

	status = clReleaseCommandQueue(commandQueue);	//Release  Command queue.
	status = clReleaseContext(context);				//Release context.
	
	sampleTimer->stopTimer(timer);
	times.releaseKernel = sampleTimer->readTimer(timer);
	

	if(!ComandArgs->verify){
		checkAgainstCpuImplementation(input, output);
	}

	freeResources();

	double SPS = ((width - 2)*(height - 2))/(times.kernelExecuting/iterations);

	cout << "Testoutput: this should be constant with different iterations!: " << (times.kernelExecuting/iterations) << endl;

	cout << "we had: " << (width - 2)*(height - 2) << " single Stancel calculations" << endl;
	cout << "this makes: \n" << SPS << " SPS (Stancels Per Second)\n" << SPS/1000 << " KSPS (Kilo Stancels Per Second)\n" << SPS/1000000 << " MSPS (Mega Stancels Per Second) \n" << SPS/1000000000 << " GSPS (Giga Stancels Per Second) \n" << endl;

	cout << "Finisched!" << endl;
	times.total= times.kernelExecuting + times.buildProgram + times.setKernelArgs + times.writeBack + times.releaseKernel;
	cout << "\nTotal time: " << times.total << endl;
	cout << "Summery times: " << endl;
	cout << "Build Program: " << times.buildProgram << endl;
	cout << "Set Kernel Args: " << times.setKernelArgs << endl;
	cout << "Kernel executon: " << times.kernelExecuting << endl;
	cout << "Get output back to host memory: " << times.writeBack << endl;
	cout << "Releasing everything: " << times.releaseKernel << endl;

	//int b;
	//cin >> b;
	return SUCCESS;
}

void freeResources(){ 

	if (output != NULL)
	{
		free(output);
		output = NULL;
	}
	if (input != NULL)
	{
		free(input);
		input = NULL;
	}
	if (devices != NULL)
	{

		free(devices);
		devices = NULL;
	}
}

void StupidCPUimplementation(float *in, float *out, int width, int height){
	for (int num = 0; num < width*height; num++){
		if (num < width || (num % width) == 0 || (num % width) == width - 1 || num >= (width*height - width)){
			out[num] = in[num];
		}
		else{
			out[num] = (in[num - 1] + in[num + 1] + in[num - width] + in[num + width])/4; //-4 * in[num] + in[num - 1] + in[num + 1] + in[num - width] + in[num + width];
		}
	}
}

int getPlatforms(void){
	/*Step1: Getting platforms and choose an available one.*/


	status = clGetPlatformIDs(0, NULL, &numPlatforms);

	if (status != CL_SUCCESS)
	{
		cout << "Error: Getting platforms!" << endl;
		return FAILURE;
	}

	fprintf(stdout, "number of platformes found: %u \n", numPlatforms);
	/*choose the an available platform. */
	if (numPlatforms > 0)
	{
		if (numPlatforms > 1){

			cout << "Witch Platform should be used?" << endl;

			cin >> PlatformToUse;
			while (PlatformToUse > numPlatforms || PlatformToUse <= 0){
				cout << "Please incert a Valid number betwen 1 and " << numPlatforms << endl;
				cin >> PlatformToUse;
			}
			cout << PlatformToUse << endl;
		}
		cl_platform_id* platforms = (cl_platform_id*)malloc(numPlatforms* sizeof(cl_platform_id));
		status = clGetPlatformIDs(numPlatforms, platforms, NULL);
		platform = platforms[PlatformToUse - 1];
		free(platforms);
	}

	return SUCCESS;
	
}

int getDevice(void){
	/*Step 2:Query the platform and choose the first GPU device if has one.Otherwise use the CPU as device.*/
	status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
	fprintf(stdout, "number of Devices found: %u\n\n", numDevices);

	devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));
	status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, numDevices, devices, NULL);
	cout << "List of Devices found:" << endl;
	cout << "\nGPUs found:" << endl;

	PrintDeviceInfo(CL_DEVICE_TYPE_GPU);

	cout << "\nCPUs found:" << endl;
	PrintDeviceInfo(CL_DEVICE_TYPE_CPU);

	cout << "\nTotal list (correct order):" << endl;

	for (j = 0; j < numDevices; j++){
		//print device name
		clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 0, NULL, &valueSize);
		value = (char*)malloc(valueSize);
		clGetDeviceInfo(devices[j], CL_DEVICE_NAME, valueSize, value, NULL);
		printf("%u: %s \n",j, value);
		free(value);
	}
	cout << "99: Stupid CPU Implementation" << endl;
	
	//if (numDevices > 1){

		cout << "\nWitch Device should be used?" << endl;

		cin >> DeviceToUse;
		cout << DeviceToUse << endl;
		while (DeviceToUse >= numDevices || DeviceToUse < 0){
			if (DeviceToUse == 99) break;
			cout << "Please incert a Valid number betwen 0 and " << numDevices - 1 << endl;
			cin >> DeviceToUse;
		}
	//}
	if (DeviceToUse == 99){
		runCpuImplementation();
		return END;
	}
	cout << "you selected device number: " << DeviceToUse << endl;

	aktiveDevice = devices + DeviceToUse;
	return SUCCESS;
}

int PrintDeviceInfo(int type){
	
	cl_uint tempNumDevices = 0;
	status = clGetDeviceIDs(platform, type, 0, NULL, &tempNumDevices);

	cl_device_id *tempDevices = (cl_device_id*)malloc(tempNumDevices * sizeof(cl_device_id));
	status = clGetDeviceIDs(platform, type, tempNumDevices, tempDevices, NULL);
	for (j = 0; j < tempNumDevices; j++){
		
		//print device name
		clGetDeviceInfo(tempDevices[j], CL_DEVICE_NAME, 0, NULL, &valueSize);
		value = (char*)malloc(valueSize);
		clGetDeviceInfo(tempDevices[j], CL_DEVICE_NAME, valueSize, value, NULL);
		printf("\nNAME: %s \n", value);
		free(value);

		//print device name
		clGetDeviceInfo(tempDevices[j], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(memsize), &memsize, NULL);
		printf("Max Clock Frequency: %u \n", memsize);

		// print parallel compute units
		clGetDeviceInfo(tempDevices[j], CL_DEVICE_MAX_COMPUTE_UNITS,
			sizeof(maxComputeUnits), &maxComputeUnits, NULL);
		printf("Parallel compute units: %d \n", maxComputeUnits);

		// print hardware device version
		clGetDeviceInfo(tempDevices[j], CL_DEVICE_VERSION, 0, NULL, &valueSize);
		value = (char*)malloc(valueSize);
		clGetDeviceInfo(tempDevices[j], CL_DEVICE_VERSION, valueSize, value, NULL);
		printf("Hardware version: %s \n", value);
		free(value);

		// print software driver version
		clGetDeviceInfo(tempDevices[j], CL_DRIVER_VERSION, 0, NULL, &valueSize);
		value = (char*)malloc(valueSize);
		clGetDeviceInfo(tempDevices[j], CL_DRIVER_VERSION, valueSize, value, NULL);
		printf("CL Driver version: %s\n", value);
		free(value);

		clGetDeviceInfo(tempDevices[j], CL_DEVICE_MAX_WORK_GROUP_SIZE, 0, NULL, &valueSize);
		value = (char*)malloc(valueSize);
		clGetDeviceInfo(tempDevices[j], CL_DEVICE_MAX_WORK_GROUP_SIZE, valueSize, value, NULL);
		printf("CL Device Max Work Group Size: %s\n", value);
		free(value);

		maxComputeUnits = 0;
		size_t maxWorkGroupSize = 0;
		maxWorkGroupSize = clGetDeviceInfo(tempDevices[j], CL_DEVICE_MAX_WORK_GROUP_SIZE,
			sizeof(maxComputeUnits), &maxComputeUnits, NULL);
		printf("Max Work Group Size: %d \n", maxComputeUnits);

		cout << "Max work Group size: " << maxWorkGroupSize << endl;

		
	}
	free(tempDevices);
	return SUCCESS;
}

int runCpuImplementation(){
	cout << "you selected the stupid cpu implementation!" << endl;

	float *inputcpu = (float*)malloc(sizeof(float) * width * height);
	fill(inputcpu, inputcpu + (width*height), 1.0);

	float *outputcpu = (float*)malloc(sizeof(float) * width * height);
	memset(outputcpu, 0, sizeof(float) * width * height);

	int timer = sampleTimer->createTimer();
	sampleTimer->resetTimer(timer);
	sampleTimer->startTimer(timer);


	for (int e = 0; e < iterations; e++){
		StupidCPUimplementation(inputcpu, outputcpu, width, height);
		StupidCPUimplementation(outputcpu, inputcpu, width, height);
	}

	sampleTimer->stopTimer(timer);
	cout << "Total time: " << sampleTimer->readTimer(timer) << endl;
	cout << "so for every run thats: " << (sampleTimer->readTimer(timer) / iterations) << endl;


	if (VERBOSE){
		cout << "Input:" << endl;
		for (int y = 0; y < height; y++){
			for (int x = 0; x < width; x++){
				cout << *(inputcpu + x + y*width)<<" ";
			}
			cout << endl;
		}
		cout << endl << "Output:" << endl;
		for (int y = 0; y < height; y++){
			for (int x = 0; x < width; x++){
				cout << *(outputcpu + x + y*width)<<" ";
			}
			cout << endl;
		}
	}

	if (output != NULL)
	{
		free(output);
		output = NULL;
	}
	if (input != NULL)
	{
		free(input);
		input = NULL;
	}
	if (outputcpu != NULL)
	{
		free(outputcpu);
		outputcpu = NULL;
	}
	if (inputcpu != NULL)
	{
		free(inputcpu);
		inputcpu = NULL;
	}
	if (devices != NULL)
	{

		free(devices);
		devices = NULL;
	}

	return SUCCESS;
}

int buildProgram(cl_program *program){
	/*Step 6: Build program. */
	status = clBuildProgram(*program, 1, aktiveDevice, NULL, NULL, NULL);
	if (status != SUCCESS){
		fprintf(stderr, "Building Program failed. error code: ");
		cout << status << endl;
		switch (status)
		{
		case CL_INVALID_PROGRAM:
			cout << "Invalid Program" << endl;
			break;
		case CL_INVALID_VALUE:
			cout << "Invalid Value" << endl;
			break;
		case CL_INVALID_DEVICE:
			cout << "Invalid Device" << endl;
			break;
		case CL_INVALID_BINARY:
			cout << "Invalid Binary" << endl;
			break;
		case CL_INVALID_BUILD_OPTIONS:
			cout << "Invalid Build Options" << endl;
			break;
		case CL_INVALID_OPERATION:
			cout << "Invalid Operation" << endl;
			break;
		case CL_COMPILER_NOT_AVAILABLE:
			cout << "Compiler Not Available" << endl;
			break;
		case CL_BUILD_PROGRAM_FAILURE:
			cout << "Build Program Failure" << endl;
			break;
		case CL_OUT_OF_HOST_MEMORY:
			cout << "Out Of Host Memory" << endl;
			break;

		default:
			cout << "Unknown Error" << endl;
			break;
		}
		return FAILURE;
	}
	else cout << "Building Programm Sucsessfull" << endl;
	return SUCCESS;
}

int readArgs(int argc, char* argv[]){

	//bool verify = false;

	ComandArgs = new CLCommandArgs();
	ComandArgs->sampleVerStr = SAMPLE_VERSION;

	// Call base class Initialize to get default configuration
	if (ComandArgs->initialize() != SDK_SUCCESS)
	{
		cout << "Read Command Arguments Failed!" << endl;
		return SDK_FAILURE;
	}
	Option* wParam = new Option;
	CHECK_ALLOCATION(wParam, "Memory Allocation error.\n");
	wParam->_sVersion = "we";
	wParam->_lVersion = "width";
	wParam->_description = "Width of matrix";
	wParam->_type = CA_ARG_INT;
	wParam->_value = &width;
	ComandArgs->AddOption(wParam);
	delete wParam;

	Option* hParam = new Option;
	CHECK_ALLOCATION(hParam, "Memory Allocation error.\n");
	hParam->_sVersion = "he";
	hParam->_lVersion = "height";
	hParam->_description = "height of matrix";
	hParam->_type = CA_ARG_INT;
	hParam->_value = &height;
	ComandArgs->AddOption(hParam);
	delete hParam;

	Option* iParam = new Option;
	CHECK_ALLOCATION(iParam, "Memory Allocation error.\n");
	iParam->_sVersion = "i";
	iParam->_lVersion = "iterations";
	iParam->_description = "number of iterations";
	iParam->_type = CA_ARG_INT;
	iParam->_value = &iterations;
	ComandArgs->AddOption(iParam);
	delete iParam;

	Option* kvParam = new Option;
	CHECK_ALLOCATION(kvParam, "Memory Allocation error.\n");
	kvParam->_sVersion = "kv";
	kvParam->_lVersion = "kernelversion";
	kvParam->_description = "witch version of the kernel shold be used";
	kvParam->_type = CA_ARG_INT;
	kvParam->_value = &kernelVersion;
	ComandArgs->AddOption(kvParam);
	delete kvParam;

	ComandArgs->parseCommandLine(argc, argv);

	cout << "\n" << width << " : " << height << " Iterations: " << iterations << endl;
	//cout << ComandArgs->verify << endl;
	cout << "using kernel version: " << kernelVersion << "\n" << endl;
}

void getKernelArgSetError(int status){
	switch (status){
		case CL_INVALID_KERNEL: 
			cout << "Invalid Kernel" << endl;
			break;
		case CL_INVALID_ARG_INDEX:
			cout << "Invalid ARG INDEX" << endl;
			break;
		case CL_INVALID_ARG_VALUE:
			cout << "Invalid ARG Value" << endl;
			break;
		case CL_INVALID_MEM_OBJECT:
			cout <<	"Invalid MEM Object" << endl;
			break;
		case CL_INVALID_SAMPLER:
			cout << "Invalid Sampler" << endl;
			break;
		case CL_INVALID_ARG_SIZE:
			cout << "Invalid ARG Size" << endl;
			break;
		default:
			cout << "Unknown Error" << endl;
			break;
	}
}

int checkAgainstCpuImplementation(float *origInput, float *clOutput){

		cout << "\nChecking result against referance cpu implementation :" << endl;

		double referanceTime = 0;

	float *inout = (float*)malloc(sizeof(float) * width * height);
	memcpy(inout, origInput, sizeof(float) * width * height);

	float *workmem = (float*)malloc(sizeof(float) * width * height);
	memset(workmem, 0, sizeof(float) * width * height);

/*
	int timer = sampleTimer->createTimer();
	sampleTimer->resetTimer(timer);
	sampleTimer->startTimer(timer);
*/
	cout << "calculateing..." << endl;
	sampleTimer->resetTimer(timer);
	sampleTimer->startTimer(timer);

	for (int e = 0; e < iterations; e++){

		StupidCPUimplementation(inout, workmem, width, height);
		StupidCPUimplementation(workmem, inout, width, height);
	}

	sampleTimer->stopTimer(timer);
	referanceTime = sampleTimer->readTimer(timer);

	if(VERBOSE){
		cout << "referance output:" << endl;
		for (int y = 0; y < height; y++){
			for (int x = 0; x < width; x++){
				cout << *(inout + x + y*width)<<" ";
			}
			cout << endl;
		}
	}

	if(memcmp(clOutput, inout , width * height * sizeof(float))==0){
		cout << "\nPassed the test; results are equil.\n" << endl;
	}
	else{
		cout << "\nFailed the test; results differ.\n" << endl;
	}

	cout << "referance took " << referanceTime << " seconds" << endl;
	cout << "so thats " << ((width - 2)*(height - 2))/referanceTime << " SPS" << endl; 
/* 
	sampleTimer->stopTimer(timer);
	cout << "Total time: " << sampleTimer->readTimer(timer) << endl;
	cout << "so for every run thats: " << (sampleTimer->readTimer(timer) / iterations) << endl;
*/


	if (workmem != NULL)
	{
		free(workmem);
		workmem = NULL;
	}
	if (inout != NULL)
	{
		free(inout);
		inout = NULL;
	}
	if (output != NULL)
	{
		free(output);
		output = NULL;
	}
	if (input != NULL)
	{
		free(input);
		input = NULL;
	}
	if (devices != NULL)
	{

		free(devices);
		devices = NULL;
	}
	return SDK_SUCCESS;
}

void getExecutionError(int status){
	switch (status){
	case CL_INVALID_WORK_ITEM_SIZE:
		cout << " Invailid work item size" << endl;
	break;
	case CL_INVALID_EVENT_WAIT_LIST:
		cout << " Invailid event wait list" << endl;
	break;
	case CL_MEM_OBJECT_ALLOCATION_FAILURE:
		cout << " mem object allocation failure" << endl;
	break;
	  
	case CL_INVALID_WORK_DIMENSION:
		cout << " Invaild work dimension" << endl;
	break;
	case CL_INVALID_PROGRAM_EXECUTABLE:
		cout << " Invalid program executable" << endl;
	break;
	case CL_INVALID_COMMAND_QUEUE:
		cout << " CL_INVALID_COMMAND_QUEUE" << endl;
	break;
	case CL_INVALID_KERNEL:
		cout << " CL_INVALID_KERNEL" << endl;
	break;
	case CL_INVALID_CONTEXT:
		cout << " CL_INVALID_CONTEXT" << endl;
	break;
	case CL_INVALID_KERNEL_ARGS:
		cout << " CL_INVALID_KERNEL_ARGS" << endl;
	break;
	case CL_INVALID_WORK_GROUP_SIZE:
		cout << " CL_INVALID_WORK_GROUP_SIZE" << endl;
	break;
	case CL_INVALID_GLOBAL_OFFSET:
		cout << " CL_INVALID_GLOBAL_OFFSET" << endl;
	break;
	case CL_OUT_OF_HOST_MEMORY:
		cout << " CL_OUT_OF_HOST_MEMORY" << endl;
	break;
	case CL_OUT_OF_RESOURCES:
		cout << " CL_OUT_OF_RESOURCES" << endl;
	break;
	default:
		cout <<" unknown error" << endl;
	break;
	}
}