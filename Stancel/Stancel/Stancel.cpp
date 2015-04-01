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


	input = (cl_float*)malloc(sizeof(cl_float) * width * height);
	fill(input, input + (width*height), 1.0);
	
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
		CL_MEM_READ_WRITE,
		sizeof(cl_float) * width * height,
		NULL,
		&status);
	if (status != SUCCESS){
		fprintf(stderr, "clCreateBuffer failed. (BufferMatrixB) %i\n", status);
		freeResources();
		return FAILURE;
	}
	
	/*Step 8: Create kernel object */
	cl_kernel kernel = clCreateKernel(program, "stancel", NULL);
	cl_kernel kernelBackwards = clCreateKernel(program, "stancel", NULL);

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
	size_t global_work_size[1] = { width * height };

	sampleTimer->resetTimer(timer);
	sampleTimer->startTimer(timer);

	for (int e = 0; e < iterations; e++){
		status = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, global_work_size, NULL, 0, NULL, &ndrEvt);
		if (status != SUCCESS) fprintf(stderr, "executing kernel failed. \n");
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
		 
		status = clEnqueueNDRangeKernel(commandQueue, kernelBackwards, 1, NULL, global_work_size, NULL, 0, NULL, &ndrEvt);
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
	}

	sampleTimer->stopTimer(timer);
	times.kernelExecuting = sampleTimer->readTimer(timer);
	cout << "Total time: " << times.kernelExecuting << endl;
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
				cout << *(input + x + y*width);
			}
			cout << endl;
		}
		cout << endl << "Output:" << endl;
		for (int y = 0; y < height; y++){
			for (int x = 0; x < width; x++){
				cout << *(output + x + y*width);
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
	

	if(ComandArgs->verify){
		checkAgainstCpuImplementation(input, output);
	}


	freeResources();
	cout << "Finisched!" << endl;
	times.total= times.kernelExecuting + times.buildProgram + times.setKernelArgs + times.writeBack + times.releaseKernel;
	cout << "\nTotal time: " << times.total << endl;
	cout << "Summery times: " << endl;
	cout << "Build Program: " << times.buildProgram << endl;
	cout << "Set Kernel Args: " << times.setKernelArgs << endl;
	cout << "Kernel executon: " << times.kernelExecuting << endl;
	cout << "Get output back to host memory: " << times.writeBack << endl;
	cout << "Ŕeleasing everything: " << times.releaseKernel << endl;

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
			out[num] = -4 * in[num] + in[num - 1] + in[num + 1] + in[num - width] + in[num + width];
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
				cout << *(inputcpu + x + y*width);
			}
			cout << endl;
		}
		cout << endl << "Output:" << endl;
		for (int y = 0; y < height; y++){
			for (int x = 0; x < width; x++){
				cout << *(outputcpu + x + y*width);
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

	ComandArgs->parseCommandLine(argc, argv);

	cout << "\n" << width << " : " << height << " Iterations: " << iterations << endl;
	cout << ComandArgs->verify << endl;
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

	for (int e = 0; e < iterations; e++){
		StupidCPUimplementation(inout, workmem, width, height);
		StupidCPUimplementation(workmem, inout, width, height);
	}

	if(VERBOSE){
		cout << "referance output:" << endl;
		for (int y = 0; y < height; y++){
			for (int x = 0; x < width; x++){
				cout << *(inout + x + y*width);
			}
			cout << endl;
		}
	}

	if(memcmp(clOutput, inout , width * height * sizeof(float))==0){
		cout << "\nPassed the test; results are equil.\n" << endl;
	}
	else{
		cout << "\njFailed the test; results differ.\n" << endl;
	}
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