// Stencil.cpp : Defines the entry point for the console application.
//
#include "Stencil.hpp"


using namespace std;

int main(int argc, char* argv[])
{

/*	KernelGenerator kGenerator;

	int test_pos[8] = {0,-1, -1,0, 1,0, 0,1};//[8] = {0,-1, -1,0, 1,0, 0,1}; 
	float test_weights[4] =  {1.0f,1.0f,1.0f,1.0f,};
	string testkernel = kGenerator.generateKernelString("teststring", test_pos, test_weights, 4, 1);
	cout << "Created Kernel: \n\n" << testkernel << "\n" << endl;
*/

	sampleTimer = new SDKTimer();
	timer = sampleTimer->createTimer();

	if(PROFILE){
		status = clInitializePerfMarkerAMD();
		if( status != AP_SUCCESS){
			if( status == AP_APP_PROFILER_NOT_DETECTED ){
				PROFILE = false;
			}
			else{
				cout << "Initilasation of App Profiler failed: " << status << endl;
				getAppProfilerInitError(status);
			}
		}
	}
	if(PROFILE){
		if(clBeginPerfMarkerAMD( "KernelProfiler", "Group1") != AP_SUCCESS){
			cout << "Begin of App Profiler failed" << endl;
		}
	}
	
	if(readArgs(argc, argv) == SDK_FAILURE){
		freeResources();
		return FAILURE;
	}


	if (getPlatforms() == FAILURE){ 
		freeResources();
		return FAILURE; 
	}
	status = getDevice();
	if (status == END){ 
		freeResources();
		return SUCCESS; 
	}else if(status == FAILURE){
		freeResources();
		return FAILURE;
	}
	
	/*Step 3: Create context.*/
	cl_context context = clCreateContext(NULL, 1, aktiveDevice, NULL, NULL, NULL);


	/*Step 4: Creating command queue associate with the context.*/
	cl_command_queue commandQueue = clCreateCommandQueue(context, *aktiveDevice, 0, NULL);

	/*Step 5: Create program object */
	sampleTimer->resetTimer(timer);
	sampleTimer->startTimer(timer);

	cl_program program;
	if(kernelVersion != 7){
		const char *filename = "Stencil_Kernel.cl";

		string KernelSource;
		status = convertToString(filename, KernelSource);
		const char *source = KernelSource.c_str();
		size_t sourceSize[] = { strlen(source) };
		if (VERBOSEKERNEL){ cout << source << endl; }
		program = clCreateProgramWithSource(context, 1, &source, sourceSize, NULL);
	}
	else{
		if(StencilDefinition.compare("default") == 0){
			cout << "no positions specifyed; using the default 5-Point-Stencil" << endl;
			positions = (cl_int*)malloc(sizeof(cl_int) * 8);
			positions[0] =  0;	positions[1] = -1;
			positions[2] = -1;	positions[3] =  0;
			positions[4] =  1;	positions[5] =  0;
			positions[6] =  0;	positions[7] =  1;
		}
		else{
			numberPoints = parseStringToPositions(StencilDefinition);
		}

		if(StencilWeights.compare("default") == 0){
			cout << "no weights specifyed; assuming there all 1.0" << endl;
			weights = (cl_float*)malloc(sizeof(cl_float) * numberPoints);
			fill(weights, weights + numberPoints, 1.0);
		}
		else{
			cl_int numberWeights = parseStringToWeights(StencilWeights);


			if(numberWeights == FAILURE){
				return FAILURE;
			}
			if(numberPoints != numberWeights){
				cout << "ERROR: number of points and number of weights differ!"<< endl;
				cout << "numberPoints = "<< numberPoints << " and numberWeights = " << numberWeights << endl;
				return FAILURE;
			}
		}
		cl_int edgewithlocal;
		/*edgewithlocal = getEdgeWidth();
		setInputEdgesToOne(edgewithlocal);
		edgewidth = edgewithlocal;
		*/
		edgewidth = getEdgeWidth();

		KernelGenerator kGenerator;

		//int test_pos[8] = {0,-1, -1,0, 1,0, 0,1};
		//float test_weights[4] =  {1.0f,1.0f,1.0f,1.0f,};
		string KernelSource = kGenerator.generateKernelString("createdKernel", positions, weights, numberPoints, edgewidth);
		//cout << "Created Kernel: \n\n" << testkernel << "\n" << endl;
		const char *source = KernelSource.c_str();
		size_t sourceSize[] = { strlen(source) };
		if (VERBOSEKERNEL){ cout << source << endl; }
		program = clCreateProgramWithSource(context, 1, &source, sourceSize, NULL);
	}

	if(buildProgram(&program) != SUCCESS) {
		freeResources();
		return FAILURE;
	}
	sampleTimer->stopTimer(timer);
	times.buildProgram = sampleTimer->readTimer(timer);

	initilizeHostBuffers();

	sampleTimer->resetTimer(timer);
	sampleTimer->startTimer(timer);
	
	/*Step 8: Create kernel object */
	cl_kernel 
	kernel = NULL, 
	kernelBackwards = NULL;

	cl_uint work_dim;
	size_t *global_work_size = (size_t*) malloc(2*sizeof(size_t));
	size_t *local_work_size = (size_t*) malloc(2*sizeof(size_t));

	status = setupKernelSpesificStuff(&work_dim, global_work_size, &local_work_size, &context,
				&kernel, &kernelBackwards, &program);
	if (status == FAILURE){
		cout << "Abording!" << endl;
		freeResources();
		return FAILURE;
	}

	sampleTimer->stopTimer(timer);
	times.setKernelArgs = sampleTimer->readTimer(timer);

	if(!ComandArgs->quiet){
		cout << "kernel Arguments are set; starting kernel now!" << endl;

		//Get maximum work goup size
		KernelWorkGroupInfo kernelInfo;
		status = kernelInfo.setKernelWorkGroupInfo(kernel, *aktiveDevice);
	    CHECK_ERROR(status,0, "setKernelWrkGroupInfo failed");
	    
    	cout << "Max kernel work gorup size: " << kernelInfo.kernelWorkGroupSize << endl;
	}

	status = runKernels(&kernel, &kernelBackwards, &commandQueue, work_dim, global_work_size, local_work_size);

	sampleTimer->resetTimer(timer);
	sampleTimer->startTimer(timer);
	/*Step 11: Read the output back to host memory.*/
	status = clEnqueueReadBuffer(commandQueue, BufferMatrixA, CL_TRUE, 0, 
		width * height * sizeof(cl_float), output, 0, NULL, NULL);

	sampleTimer->stopTimer(timer);
	times.writeBack = sampleTimer->readTimer(timer);

	if (VERBOSEMATRIX){
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
	if(!ComandArgs->quiet){
		cout << "done; releasinng kernel now" << endl;
	}
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
	
	if(PROFILE){
		if( clEndPerfMarkerAMD() != AP_SUCCESS){
			cout << "Error ending Profiler" << endl;
		}
		if( clFinalizePerfMarkerAMD() != AP_SUCCESS){
			cout << "Error ending Profiler" << endl;
		}

	}

	if(!ComandArgs->verify){
		checkAgainstCpuImplementation(input, output);
	}

	freeResources();
	printStats();

	return SUCCESS;
}

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
	if (positions != NULL)
	{
		free(positions);
		positions = NULL;
	}
	if (weights != NULL)
	{
		free(weights);
		weights = NULL;
	}
}

void StupidCPUimplementation(cl_float *in, cl_float *out, int width, int height){
	for (int num = 0; num < width*height; num++){
		if (num < width || (num % width) == 0 || (num % width) == width - 1 || num >= (width*height - width)){
			out[num] = in[num];
		}
		else{
			out[num] = (in[num - 1] + in[num + 1] + in[num - width] + in[num + width])/4; 
			//-4 * in[num] + in[num - 1] + in[num + 1] + in[num - width] + in[num + width];
		}
	}
}
void StupidDynamicCPUImplementation(cl_float *in, cl_float *out, 
					int width, int height, cl_int *positions, cl_float *allWeights, 
					cl_int numberPoints){

	cl_float sum;
	int lookAt;
	float sumOfWeights = 0;

	for(int e = 0; e < numberPoints; e++){
		sumOfWeights += fabs(allWeights[e]); 
	}
	
	for (int num = 0; num < width*height; num++){

		if(num/width < edgewidth || num/width > height - edgewidth - 1){
			continue;
		}
		if(num%width < edgewidth || num%width > width - edgewidth - 1){
			continue;	
		}

		sum = 0;
		lookAt = 0;
		for (int i = 0; i < numberPoints*2; i = i + 2){
			lookAt = num + positions[i] + positions[i+1]*width;

			sum += in[lookAt] * allWeights[i/2];
		}
		
		out[num] = sum/sumOfWeights;
	}
}

/*Step1: Getting platforms and choose an available one.*/
int getPlatforms(void){

	status = clGetPlatformIDs(0, NULL, &numPlatforms);

	if (status != CL_SUCCESS)
	{
		cout << "Error: Getting platforms!" << endl;
		return FAILURE;
	}
	if(!ComandArgs->quiet){
		cout << "number of platformes found: " << numPlatforms << endl;
	}
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

/*Step 2:Query the platform and choose the first GPU device if has one.Otherwise use the CPU as device.*/
int getDevice(void){
	if(device.compare("cpu") == 0){
		status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0, NULL, &numDevices);
		if(!ComandArgs->quiet){
			cout << "\nusing CPU; found: " << numDevices << " device(s)" << endl;
		}
		devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));
		status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, numDevices, devices, NULL);
	}
	else if(device.compare("stupid") == 0){
		if(!ComandArgs->quiet){
			cout << "\nyou selected the stupid CPU implementation" << endl;
		}
		runCpuImplementation();
		return END;
	}
	else{
		status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
		if(numDevices == 0){
			if(!ComandArgs->quiet){
				cout << "\nNo GPU was found; falling back to CPU" << endl;
			}
			status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0, NULL, &numDevices);
			if(!ComandArgs->quiet){
				cout << "using CPU; found: " << numDevices << " device(s)" << endl;
			}
			devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));
			status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, numDevices, devices, NULL);
		}else{
			if(!ComandArgs->quiet){
				cout << "\nusing GPU; found: " << numDevices << " device(s)" << endl;
			}
			devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));
			status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);
		}
	}

	if(VERBOSE){
		cout << "List of all Devices found:" << endl;
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
	}

	if(numDevices == 0){
		cout << "Fatal error; no device was found" << endl;
		return FAILURE;
	}
	if(!ComandArgs->quiet){
		cout << "\nusing: " << endl;
		clGetDeviceInfo(devices[0], CL_DEVICE_NAME, 0, NULL, &valueSize);
			value = (char*)malloc(valueSize);
			clGetDeviceInfo(devices[0], CL_DEVICE_NAME, valueSize, value, NULL);
			printf("%s \n\n", value);
			free(value);
	}
	aktiveDevice = &devices[0];
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
	srand (static_cast <unsigned> (time(0)));

	float *inputcpu = (float*)malloc(sizeof(float) * width * height);
	fill(inputcpu, inputcpu + (width*height), 1.0);

	for (int i = width; i < (width*height)-width; i++){
		if(i%width == 0 || i%width == (width-1)){
			continue;
		}
		inputcpu[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/10));
	}
	
	float *outputcpu = (float*)malloc(sizeof(float) * width * height);
	memcpy(outputcpu, inputcpu, sizeof(float) * width * height);

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

/*Step 6: Build program. */
int buildProgram(cl_program *program){
	status = clBuildProgram(*program, 1, aktiveDevice, NULL, NULL, NULL);
	
	if (status != SUCCESS){
		buildProgramError(status);
		return FAILURE;
	}
	else{
		if(!ComandArgs->quiet){
	 		cout << "Building Programm Sucsessfull" << endl;
		}
	}
	if(!ComandArgs->quiet || status != SUCCESS){
		// Get log information
	    size_t log_size;
	    clGetProgramBuildInfo(*program, *aktiveDevice, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

	    char *log; 
	    log = (char *) malloc(log_size);

	    clGetProgramBuildInfo(*program, *aktiveDevice, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

	    printf("%s\n", log);
	    free(log);
	}

	return SUCCESS;
}

int readArgs(int argc, char* argv[]){

	ComandArgs = new CLCommandArgs();
	ComandArgs->sampleVerStr = SAMPLE_VERSION;

	// Call base class Initialize to get default configuration
	if (ComandArgs->initialize() != SDK_SUCCESS)
	{
		cout << "Read Command Arguments Failed!" << endl;
		return SDK_FAILURE;
	}
	// delete default Options I don't need
	Option* tParam = new Option;
	CHECK_ALLOCATION(tParam, "Memory Allocation error.\n");
	tParam->_sVersion = "t";
	ComandArgs->DeleteOption(tParam);
	tParam->_sVersion = "p";	
	ComandArgs->DeleteOption(tParam);
	tParam->_sVersion = "d";
	ComandArgs->DeleteOption(tParam);
	tParam->_sVersion = "v";
	ComandArgs->DeleteOption(tParam);
	delete tParam;

	Option* longParam = new Option;
	CHECK_ALLOCATION(longParam, "Memory Allocation error.\n");
	longParam->_lVersion = "dump";
	ComandArgs->DeleteOption(longParam);
	longParam->_lVersion = "load";
	ComandArgs->DeleteOption(longParam);
	longParam->_lVersion = "flags";
	ComandArgs->DeleteOption(longParam);
	delete longParam;
	

	// add special Options I do ned
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

	Option* stParam = new Option;
	CHECK_ALLOCATION(stParam, "Memory Allocation error.\n");
	stParam->_sVersion = "st";
	stParam->_lVersion = "Stencil";
	stParam->_description = "define what kind of stanil should be used";
	stParam->_type = CA_ARG_STRING;
	stParam->_value = &StencilDefinition;
	ComandArgs->AddOption(stParam);
	delete stParam;

	Option* stwParam = new Option;
	CHECK_ALLOCATION(stwParam, "Memory Allocation error.\n");
	stwParam->_sVersion = "w";
	stwParam->_lVersion = "weights";
	stwParam->_description = "define weights for every point in the dynamic Stencil";
	stwParam->_type = CA_ARG_STRING;
	stwParam->_value = &StencilWeights;
	ComandArgs->AddOption(stwParam);
	delete stwParam;

	Option* deviceParam = new Option;
	CHECK_ALLOCATION(deviceParam, "Memory Allocation error.\n");
	deviceParam->_sVersion = "d";
	deviceParam->_lVersion = "device";
	deviceParam->_description = "define what kind of device should be used";
	deviceParam->_type = CA_ARG_STRING;
	deviceParam->_value = &device;
	deviceParam->_usage = " [gpu | cpu | all | stupid]";
	ComandArgs->AddOption(deviceParam);
	delete deviceParam;

	Option* verboseParam = new Option;
	CHECK_ALLOCATION(verboseParam, "Memory Allocation error.\n");
	verboseParam->_sVersion = "V";
	verboseParam->_lVersion = "verbose";
	verboseParam->_description = "get more output";
	verboseParam->_type = CA_NO_ARGUMENT;
	verboseParam->_value = &VERBOSE;
	ComandArgs->AddOption(verboseParam);
	delete verboseParam;

	Option* verbosemaParam = new Option;
	CHECK_ALLOCATION(verbosemaParam, "Memory Allocation error.\n");
	verbosemaParam->_sVersion = "VM";
	verbosemaParam->_lVersion = "verbosematirx";
	verbosemaParam->_description = "print out calculated matrix";
	verbosemaParam->_type = CA_NO_ARGUMENT;
	verbosemaParam->_value = &VERBOSEMATRIX;
	ComandArgs->AddOption(verbosemaParam);
	delete verbosemaParam;

	Option* verbosekeParam = new Option;
	CHECK_ALLOCATION(verbosekeParam, "Memory Allocation error.\n");
	verbosekeParam->_sVersion = "VK";
	verbosekeParam->_lVersion = "verbosekernel";
	verbosekeParam->_description = "print out kernel file";
	verbosekeParam->_type = CA_NO_ARGUMENT;
	verbosekeParam->_value = &VERBOSEKERNEL;
	ComandArgs->AddOption(verbosekeParam);
	delete verbosekeParam;

	ComandArgs->parseCommandLine(argc, argv);

	if(!ComandArgs->quiet){
		cout << "\n" << width << " : " << height << " Iterations: " << iterations << endl;
		cout << "using kernel version: " << kernelVersion << "\n" << endl;
	}

	return SUCCESS;
}

int checkAgainstCpuImplementation(cl_float *origInput, cl_float *clOutput){
	if(!ComandArgs->quiet){
		cout << "\nChecking result against referance cpu implementation :" << endl;
	}
	double referanceTime = 0;

	cl_float *inout = (cl_float*)malloc(sizeof(cl_float) * width * height);
	memcpy(inout, origInput, sizeof(cl_float) * width * height);

	cl_float *workmem = (cl_float*)malloc(sizeof(cl_float) * width * height);
	//memset(workmem, 0, sizeof(cl_float) * width * height);
	memcpy(workmem, origInput, sizeof(cl_float) * width * height);


/*
	int timer = sampleTimer->createTimer();
	sampleTimer->resetTimer(timer);
	sampleTimer->startTimer(timer);
*/
	if(!ComandArgs->quiet){
		cout << "calculateing..." << endl;
	}
	sampleTimer->resetTimer(timer);
	sampleTimer->startTimer(timer);

	if(kernelVersion == 6 || kernelVersion == 7){
		for (int e = 0; e < iterations; e++){
			StupidDynamicCPUImplementation(inout, workmem, width, height, positions, weights, numberPoints);
			StupidDynamicCPUImplementation(workmem, inout, width, height, positions, weights, numberPoints);
		}
	}
	else{
		for (int e = 0; e < iterations; e++){

			StupidCPUimplementation(inout, workmem, width, height);
			StupidCPUimplementation(workmem, inout, width, height);
		}
	}
	sampleTimer->stopTimer(timer);
	referanceTime = sampleTimer->readTimer(timer);

	if(VERBOSEMATRIX){
		cout << "referance output:" << endl;
		for (int y = 0; y < height; y++){
			for (int x = 0; x < width; x++){
				cout << *(inout + x + y*width)<<" ";
			}
			cout << endl;
		}
	}

	if(memcmp(clOutput, inout , width * height * sizeof(cl_float))==0){
		cout << "\nPassed the test; results are equil.\n" << endl;
	}
	else{
		cout << "\nFailed the test; results differ." << endl;
		if(chekMemSimilar(clOutput, inout, width * height ) == 0){
			cout <<" But Results are similar; should be ok! \n" << endl;
		}
		else{
			cout <<" Not even close! \n" << endl;
		}
	}

	cout << "referance took " << referanceTime << " seconds" << endl;
	cout << "so thats " << ((width - 2)*(height - 2))/(referanceTime/iterations) << " SPS" << endl; 
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

int chekMemSimilar(cl_float* openCl, cl_float* referance, int length){
	cl_float test;
	float maxDiff = 0, maxprocent = 0;
	for(int i = 0; i < length;i++){
		test = openCl[i] - referance[i];
			if(abs(test) > maxDiff){
				maxDiff = abs(test);
				maxprocent = (abs(test)/abs(max(openCl[i],referance[i])))*100;
			}
	}
	cout << "Max difference is: "<< maxDiff << endl;
	cout << "In Procent: " << maxprocent << "%" << endl;
	if(maxprocent > 0.0001f){
		return -1;
	}
	return 0;
}

cl_int parseStringToPositions(std::string str){
	str.erase(std::remove(str.begin(), str.end(), ' ') 
		,str.end());

	cl_int *helper = (cl_int*)malloc(sizeof(cl_int) * str.size());

	cl_int i = 0, e = 0;
	bool negativ;
	int result;

	while(i < str.size()){
		negativ = false;
		result = 0;

		while(str[i] != ','){
			if (str[i] == '-'){
				negativ = true;
				i++;
				continue;
			}
			result = result * 10;
			result += str[i] - '0';
			i++;
			if(i == str.size()){
				break;
			}
		}
		if (negativ)	result = -result;

		//cout << "single number is: " << result <<"and i: " << i << endl;

		helper[e] = result;
		e++;
		i++;
	}



	positions = (cl_int*)malloc(sizeof(cl_int) * e);

	memcpy(positions, helper, sizeof(cl_int) * e);

	free(helper);

	//cout << "first 4 numbers are:: " << positions[0] << " " << positions[1] << " " 
									//<< positions[2] << " " << positions[3] << endl;
	return e/2;
}

cl_int parseStringToWeights(std::string str){

//	cout << " using String for weights: " << str << endl;
	str.erase(std::remove(str.begin(), str.end(), ' ') 
		,str.end());

//	cout << " String wihtout witespaces: " << str << endl;

	
	cl_float *temp = (cl_float*)malloc(sizeof(cl_float) * str.size());

	cl_int i = 0, e = 0, dotcount;
	bool negativ, dot;
	cl_float result;

	while(i < str.size()){
		negativ = dot = false;
		result = 0;
		dotcount = 0;

		while(str[i] != ','){
			if (str[i] == '-'){
				negativ = true;
				i++;
				continue;
			}
			if (str[i] == '.'){
				if(dot == true){
					cout << "ERROR while pasing weights string: number with "
								<< "multiple dots; maybe you missed a ," << endl;
					return FAILURE;
				} 
				dot = true;
				i++;
				continue;
			}
			result = result * 10;
			result += str[i] - '0';
			i++;
			if(dot) dotcount++;
			if(i == str.size()){
				break;
			}
		}
		if (dot){
		//	cout << "result " << result << " 10^dotcount " << (pow(10,dotcount)) 
							//<< " dotcount: " << dotcount << endl;
			result = result/ pow(10,dotcount);	
		} 
		if (negativ)	result = -result;

//		cout << "single number is: " << result <<"and i: " << i << endl;

		temp[e] = result;
		e++;
		i++;
	}
	weights = (cl_float*)malloc(sizeof(cl_float) * e);

	memcpy(weights, temp, sizeof(cl_float) * e);

	free(temp);
//	cout << " nubmer of weights: "<< e << endl;
//	cout << " first 4 numbers are: " << weights[0] << " " << weights[1] << " " 
//									 << weights[2] << " " << weights[3] << endl;

	return e;
}

int setupKernelSpesificStuff(cl_uint* work_dim, size_t *global_work_size, size_t **local_work_size, 
				cl_context* context, cl_kernel* kernel, cl_kernel* kernelBackwards, cl_program* program){
float sumOfWeights = 0;

	switch (kernelVersion){
		case 1:
			*kernel = clCreateKernel(*program, "Stencil1", NULL);
			*kernelBackwards = clCreateKernel(*program, "Stencil1", NULL);

			status = setBufferKernelArgs(kernel, kernelBackwards, context);
			if(status != SUCCESS){
				cout << "Abording!" << endl;
				freeResources();
				return FAILURE;
			}

			*work_dim = 1;
			global_work_size[0] = width * height;
			*local_work_size = NULL;
		break;
		case 2:
			*kernel = clCreateKernel(*program, "Stencil2", NULL);
			*kernelBackwards = clCreateKernel(*program, "Stencil2", NULL);

			status = setBufferKernelArgs(kernel, kernelBackwards, context);
			if(status != SUCCESS){
				cout << "Abording!" << endl;
				freeResources();
				return FAILURE;
			}

			*work_dim = 2;
			global_work_size[0] = width - 2;
			global_work_size[1] = height - 2;
			*local_work_size = NULL;
		break;
		case 3:
			*kernel = clCreateKernel(*program, "Stencil3", NULL);
			*kernelBackwards = clCreateKernel(*program, "Stencil3", NULL);

			status = setBufferKernelArgs(kernel, kernelBackwards, context);
			if(status != SUCCESS){
				cout << "Abording!" << endl;
				freeResources();
				return FAILURE;
			}

			*work_dim = 2;
			global_work_size[0] = (width - 2);
			global_work_size[1] = (height - 2);
			(*local_work_size)[0] = 4;
			(*local_work_size)[1] = 4;
			
			for (int i = min(global_work_size[0], (size_t) 16); i > 0; i--)		
			{
				//(size_t)(sqrt(kernelInfo.kernelWorkGroupSize)) in min
					if(global_work_size[0]%i == 0){
					(*local_work_size)[0] = (*local_work_size)[1] = i;
					break; 
				}
			}
			if(!ComandArgs->quiet){
				cout << "Using blocks of size: " << (*local_work_size)[0] <<" ; "<< (*local_work_size)[1] << endl;
			}
			// Create local mem objects to cash blocks in 
			status = clSetKernelArg(*kernel, 4, ((*local_work_size)[0] + 2) * ((*local_work_size)[1] + 2) * sizeof(cl_float), NULL);
		    CHECK_OPENCL_ERROR(status, "clSetKernelArg failed. (local memory)");

		    status = clSetKernelArg(*kernelBackwards, 4, ((*local_work_size)[0] + 2) * ((*local_work_size)[1] + 2) * sizeof(cl_float), NULL);
		    CHECK_OPENCL_ERROR(status, "clSetKernelArg failed. (local memory)");

			if(!ComandArgs->quiet){
				cout <<" height  and    width     "<< height << " " << width << endl;

				cout <<" global work size:  we ; he   "<< global_work_size[0] <<" ; "<< global_work_size[1] << endl;

				cout <<" lokal work size:  we ; he   "<< (*local_work_size)[0] <<" ; " << (*local_work_size)[1] << endl;
			}
		break;
		case 4:
			*kernel = clCreateKernel(*program, "Stencil4", NULL);
			*kernelBackwards = clCreateKernel(*program, "Stencil4", NULL); 

			status = setBufferKernelArgs(kernel, kernelBackwards, context);
			if(status != SUCCESS){
				cout << "Abording!" << endl;
				freeResources();
				return FAILURE;
			}

			//*work_dim = 2;
			*work_dim = 1;	
			global_work_size[0] = (width - 2);
			//global_work_size[1] = 4;
			(*local_work_size)[0] = min((cl_uint)64,(width-2)); //64
			//(*local_work_size)[1] = 4;
			if(!ComandArgs->quiet){
				cout <<" height  and    width     "<< height << " " << width << endl;

				cout <<" global work size:  we ; he   "<< global_work_size[0] <<" ; "<< global_work_size[1] << endl;
	 
				cout <<" lokal work size:  we ; he   "<< (*local_work_size)[0] <<" ; " << (*local_work_size)[1] << endl;
			}
		break;
		case 5:

			cout << "Kernel 5 not available at the moment; sorry" << endl;
				freeResources();
				return FAILURE;


			*kernel = clCreateKernel(*program, "Stencil4_1", NULL);
			*kernelBackwards = clCreateKernel(*program, "Stencil4_1", NULL); 

			status = setBufferKernelArgs(kernel, kernelBackwards, context);
			if(status != SUCCESS){
				cout << "Abording!" << endl;
				freeResources();
				return FAILURE;
			}

			//work_dim = 2;
			*work_dim = 1;
			global_work_size[0] = (width - 2);
			//global_work_size[1] = 0;
			(*local_work_size)[0] = min((cl_uint)64,(width-2));
			//(*local_work_size)[1] = 0;

			status = clSetKernelArg(*kernel, 4, ((*local_work_size)[0] + 2) * sizeof(cl_float), NULL);
		    CHECK_OPENCL_ERROR(status, "clSetKernelArg failed. (local memory1)");

			status = clSetKernelArg(*kernel, 5, ((*local_work_size)[0] + 2) * sizeof(cl_float), NULL);
		    CHECK_OPENCL_ERROR(status, "clSetKernelArg failed. (local memory2)");

		    status = clSetKernelArg(*kernel, 6, ((*local_work_size)[0] + 2) * sizeof(cl_float), NULL);
		    CHECK_OPENCL_ERROR(status, "clSetKernelArg failed. (local memory3)");

		    status = clSetKernelArg(*kernel, 7, ((*local_work_size)[0] + 2) * sizeof(cl_float), NULL);
		    CHECK_OPENCL_ERROR(status, "clSetKernelArg failed. (local memory4)");

		    status = clSetKernelArg(*kernel, 8, sizeof(cl_float *), NULL);
		    CHECK_OPENCL_ERROR(status, "clSetKernelArg failed. (local memory4.1)");



		    status = clSetKernelArg(*kernelBackwards, 4, ((*local_work_size)[0] + 2) * sizeof(cl_float), NULL);
		    CHECK_OPENCL_ERROR(status, "clSetKernelArg failed. (local memory5)");

		    status = clSetKernelArg(*kernelBackwards, 5, ((*local_work_size)[0] + 2) * sizeof(cl_float), NULL);
		    CHECK_OPENCL_ERROR(status, "clSetKernelArg failed. (local memory6)");

		    status = clSetKernelArg(*kernelBackwards, 6, ((*local_work_size)[0] + 2) * sizeof(cl_float), NULL);
		    CHECK_OPENCL_ERROR(status, "clSetKernelArg failed. (local memory7)");

		    status = clSetKernelArg(*kernelBackwards, 7, ((*local_work_size)[0] + 2) * sizeof(cl_float), NULL);
		    CHECK_OPENCL_ERROR(status, "clSetKernelArg failed. (local memory8)");

		    status = clSetKernelArg(*kernelBackwards, 8, sizeof(cl_float *), NULL);
		    CHECK_OPENCL_ERROR(status, "clSetKernelArg failed. (local memory8.1)");



 			if(!ComandArgs->quiet){
				cout <<" height  and    width     "<< height << " " << width << endl;

				cout <<" global work size:  we "<< global_work_size[0] << endl;
	 
				cout <<" lokal work size:  we "<< (*local_work_size)[0] << endl;
			}
		break;
		case 6:
		
			*kernel = clCreateKernel(*program, "dynamicStencil1", NULL);
			*kernelBackwards = clCreateKernel(*program, "dynamicStencil1", NULL); 
			
			if(StencilDefinition.compare("default") == 0){
				cout << "no positions specifyed; using the default 5-Point-Stencil" << endl;
				positions = (cl_int*)malloc(sizeof(cl_int) * 8);
				positions[0] =  0;	positions[1] = -1;
				positions[2] = -1;	positions[3] =  0;
				positions[4] =  1;	positions[5] =  0;
				positions[6] =  0;	positions[7] =  1;

			//	0,-1, -1,0, 1,0, 0,1
			}
			else{
				numberPoints = parseStringToPositions(StencilDefinition);
			}

			if(StencilWeights.compare("default") == 0){
				cout << "no weights specifyed; assuming there all 1.0" << endl;
				weights = (cl_float*)malloc(sizeof(cl_float) * numberPoints);
				fill(weights, weights + numberPoints, 1.0);
			}
			else{
				cl_int numberWeights = parseStringToWeights(StencilWeights);


				if(numberWeights == FAILURE){
					return FAILURE;
				}
				if(numberPoints != numberWeights){
					cout << "ERROR: number of points and number of weights differ!"<< endl;
					cout << "numberPoints = "<< numberPoints << " and numberWeights = " << numberWeights << endl;
					return FAILURE;
				}
			}
			cl_int edgewithlocal;
			edgewithlocal = getEdgeWidth();
			setInputEdgesToOne(edgewithlocal);
			edgewidth = edgewithlocal;

			
			for(int e = 0; e < numberPoints; e++){
				sumOfWeights += fabs(weights[e]); 
			}

			status = setBufferKernelArgs(kernel, kernelBackwards, context);
			if(status != SUCCESS){
				cout << "Abording!" << endl;
				freeResources();
				return FAILURE;
			}

			*work_dim = 2;
			global_work_size[0] = width - 2*edgewidth;
			global_work_size[1] = height - 2*edgewidth;
			//free(local_work_size);
			*local_work_size = NULL;//min((cl_uint)64,(width-2));

			

			cl_mem BufferPositions;
			BufferPositions = clCreateBuffer(
				*context,
				CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
				sizeof(cl_int) * numberPoints*2,
				positions,
				&status);
			if (status != SUCCESS){
				fprintf(stderr, "clCreateBuffer failed. (BufferPositions) %i\n VGL: %i numberPoints: %i \n", 
					status, CL_INVALID_BUFFER_SIZE, numberPoints);
				freeResources();
				return FAILURE;
			}

			cl_mem BufferWeights;
			BufferWeights = clCreateBuffer(
				*context,
				CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
				sizeof(cl_float) * numberPoints,
				weights,
				&status);
			if (status != SUCCESS){
				fprintf(stderr, "clCreateBuffer failed. (BufferWeights) %i\nVGL: %i numberPoints: %i \n", 
														status,CL_INVALID_HOST_PTR, numberPoints);
				freeResources();
				return FAILURE;
			}

			status = clSetKernelArg(*kernel, 4, sizeof(cl_mem), (void *)&BufferPositions);
			status = clSetKernelArg(*kernel, 5, sizeof(cl_mem), (void *)&BufferWeights);
			status = clSetKernelArg(*kernel, 6, sizeof(cl_int), (void *)&numberPoints);
			status = clSetKernelArg(*kernel, 7, sizeof(cl_int), (void *)&sumOfWeights);
			status = clSetKernelArg(*kernel, 8, sizeof(cl_int), (void *)&edgewidth);

			status = clSetKernelArg(*kernelBackwards, 4, sizeof(cl_mem), (void *)&BufferPositions);
			status = clSetKernelArg(*kernelBackwards, 5, sizeof(cl_mem), (void *)&BufferWeights);
			status = clSetKernelArg(*kernelBackwards, 6, sizeof(cl_int), (void *)&numberPoints);
			status = clSetKernelArg(*kernelBackwards, 7, sizeof(cl_int), (void *)&sumOfWeights);
			status = clSetKernelArg(*kernelBackwards, 8, sizeof(cl_int), (void *)&edgewidth);
			
			if(VERBOSE){
				cout <<" working dimension: " << *work_dim << endl;
				cout <<" height  and    width     "<< height << " " << width << endl;

				cout <<" global work size:  we; he "<< global_work_size[0]<< global_work_size[1] << endl;
	 
				cout <<" lokal work size:  we "<< *local_work_size << endl;
			}
		break;
		case 7:

			setInputEdgesToOne(edgewidth);
			
			*kernel = clCreateKernel(*program, "createdKernel", NULL);
			*kernelBackwards = clCreateKernel(*program, "createdKernel", NULL);

			status = setBufferKernelArgs(kernel, kernelBackwards, context);
			if(status != SUCCESS){
				cout << "Abording!" << endl;
				freeResources();
				return FAILURE;
			}


			*work_dim = 2;
			global_work_size[0] = width - 2*edgewidth;
			global_work_size[1] = height - 2*edgewidth;
			*local_work_size = NULL;

		break;
		default:
			cout << "FATAL ERROR: specifyed kernel version does not exist! ("<< kernelVersion <<")"<< endl;
			return FAILURE;

	}
	return SUCCESS;

}

cl_int getEdgeWidth(){

	cl_int maximum = 0;
	for (int i = 0; i < numberPoints*2; i++){
		if(abs(positions[i]) > maximum){
			maximum = abs(positions[i]);
		}
	}
	return maximum;
}

void setInputEdgesToOne(cl_int edgewidth){
	if(edgewidth == 1){return;}
	for (int i = 0; i < height*width; i++){
		if(i/width < edgewidth || i/width > height - edgewidth - 1){
			input[i] = 1;
			output[i] = 1;
		}
		else if(i%width < edgewidth || i%width > width - edgewidth - 1){
			input[i] = 1;
			output[i] = 1;
		}
	}
}

void initilizeHostBuffers(){
	srand (static_cast <unsigned> (time(0)));
	
	input = (cl_float*)malloc(sizeof(cl_float) * width * height);
	fill(input, input + (width*height), 1.0);
	//input[5] = 10;
	//input[14] = 5;
	//input[29] = 50;
	//input[30] = 4;
	//input[25] = 10;
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
	//memset(output, 0, sizeof(cl_float) * width * height);
	memcpy(output, input, sizeof(cl_float) * width * height);

	if(VERBOSEMATRIX){
	cout << "Initional Input Matrix:" << endl;
		for (int y = 0; y < height; y++){
			for (int x = 0; x < width; x++){
				cout << *(input + x + y*width)<<" ";
			}
			cout << endl;
		}
	}
}

int setBufferKernelArgs(cl_kernel* kernel, cl_kernel* kernelBackwards, cl_context* context){

	// Create buffer for matrix A
	BufferMatrixA = clCreateBuffer(
		*context,
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
	BufferMatrixB = clCreateBuffer(
		*context,
		CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
		sizeof(cl_float) * width * height,
		input,
		&status);
	if (status != SUCCESS){
		fprintf(stderr, "clCreateBuffer failed. (BufferMatrixB) %i\n", status);
		freeResources();
		return FAILURE;
	}
	

	/*Step 9: Sets Kernel arguments.*/
	status = clSetKernelArg(*kernel, 0, sizeof(cl_mem), (void *)&BufferMatrixA);
	status = clSetKernelArg(*kernel, 1, sizeof(cl_mem), (void *)&BufferMatrixB);
	status = clSetKernelArg(*kernel, 2, sizeof(cl_int), (void *)&width);
	status = clSetKernelArg(*kernel, 3, sizeof(cl_int), (void *)&height);
	if (status != SUCCESS){	
		fprintf(stderr, "setting kernel arguments failed. \n");	
		cout << status << endl;
		getKernelArgSetError(status);
		freeResources();
		return FAILURE;
	}


	status = clSetKernelArg(*kernelBackwards, 0, sizeof(cl_mem), (void *)&BufferMatrixB);
	status = clSetKernelArg(*kernelBackwards, 1, sizeof(cl_mem), (void *)&BufferMatrixA);
	status = clSetKernelArg(*kernelBackwards, 2, sizeof(cl_int), (void *)&width);
	status = clSetKernelArg(*kernelBackwards, 3, sizeof(cl_int), (void *)&height);
	if (status != SUCCESS){	
		fprintf(stderr, "setting kernelBackwards arguments failed. \n");
		cout << status << endl;
		getKernelArgSetError(status);
		freeResources();
		return FAILURE;
	}
	return SUCCESS;
}


int runKernels(cl_kernel* kernel, cl_kernel* kernelBackwards, cl_command_queue* commandQueue,
						 size_t work_dim, size_t *global_work_size, size_t *local_work_size){
	sampleTimer->resetTimer(timer);
	sampleTimer->startTimer(timer);
	int timer2 = sampleTimer->createTimer();

	double timer2value = 0;

		if(SINGLETIME && !ComandArgs->quiet){ 
			cout << "Time \t\t SPS " << endl;
			} 

	for (int e = 0; e < iterations; e++){
		if(SINGLETIME){
			sampleTimer->resetTimer(timer2);
			sampleTimer->startTimer(timer2);
		}

		status = clEnqueueNDRangeKernel(*commandQueue, *kernel, work_dim, NULL, 
								global_work_size, local_work_size, 0, NULL, &ndrEvt);
		if (status != SUCCESS){
			fprintf(stderr, "executing kernel failed. %i \n ",status); 
			getExecutionError(status);
			freeResources();
			return FAILURE;
		}
		status = clFlush(*commandQueue);

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
		
		status = clEnqueueNDRangeKernel(*commandQueue, *kernelBackwards, work_dim, NULL, 
										global_work_size, local_work_size, 0, NULL, &ndrEvt);
		if (status != SUCCESS) fprintf(stderr, "executing kernel simply just for the second try failed. \n");
		status = clFlush(*commandQueue);

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
			if(!ComandArgs->quiet){
				cout << timer2value  << " \t " << ((width - 2)*(height - 2))/ timer2value << endl;
			}
		}
	}

	sampleTimer->stopTimer(timer);
	times.kernelExecuting = sampleTimer->readTimer(timer);
	if(!ComandArgs->quiet){
		cout << "Total executing time: " << times.kernelExecuting << endl;
		cout << "so for every run thats: " << (sampleTimer->readTimer(timer) / iterations) << endl;
	}

	return SUCCESS;

}

void printStats(){

	double SPS = ((width - 2)*(height - 2))/(times.kernelExecuting/iterations);

	//cout << "Testoutput: this should be constant with different iterations!: " 
								//<< (times.kernelExecuting/iterations) << endl;
	if(!ComandArgs->quiet){
		cout << "we had: " << (width - 2)*(height - 2) << " single Stencil calculations" << endl;
		cout << "this makes: ";
	}
	cout <<"\n"<< SPS << " SPS (Stencils Per Second)\n" << SPS/1000 << " KSPS (Kilo Stencils Per Second)\n" << SPS/1000000 
			<< " MSPS (Mega Stencils Per Second) \n" << SPS/1000000000 << " GSPS (Giga Stencils Per Second) \n" << endl;
	if(!ComandArgs->quiet){
		cout << "Finisched!" << endl;
	}
	times.total= times.kernelExecuting + times.buildProgram + times.setKernelArgs + times.writeBack + times.releaseKernel;
	cout << "\nTotal time: " << times.total << endl;
	cout << "Summery times: " << endl;
	cout << "Build Program: " << times.buildProgram << endl;
	cout << "Set Kernel Args: " << times.setKernelArgs << endl;
	cout << "Kernel executon: " << times.kernelExecuting << endl;
	cout << "Get output back to host memory: " << times.writeBack << endl;
	cout << "Releasing everything: " << times.releaseKernel << endl;

}