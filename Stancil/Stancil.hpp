
#include <CL/cl.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <fstream>
#include <ctime>
#include "CLUtil.hpp"
#include <algorithm>
#include "error_handling_methods.cpp"
#include "CLPerfMarker.h"


using namespace appsdk;

#define SUCCESS 0
#define FAILURE 1 
#define END 99


#define SINGLETIME true 
#define PROFILE true

#define WIDTH  500
#define HEIGHT 500
#define SAMPLE_VERSION "sample"
#define ITERATIONS 500

cl_uint numPlatforms = 0;	//the NO. of platforms
cl_platform_id platform = NULL;	//the chosen platform
cl_int	status = 0;
int PlatformToUse = 1;

cl_uint				numDevices = 0;
cl_device_id        *devices;
cl_device_id		*aktiveDevice;

cl_mem BufferMatrixA;
cl_mem BufferMatrixB;

size_t valueSize;
char* value;
cl_uint memsize, j;
cl_uint maxComputeUnits;
cl_uint DeviceToUse = 0;
cl_uint width = WIDTH, height = HEIGHT;
cl_uint iterations = ITERATIONS;
cl_uint kernelVersion = 3;
std::string stancilDefinition = "default";
std::string stancilWeights = "default";
cl_int edgewith = 1;
std::string device = "gpu";
bool VERBOSE = false;
bool VERBOSEMATRIX = false;
bool VERBOSEKERNEL = false;

cl_float *input = NULL;
cl_float *output = NULL;

cl_int numberPoints = 4;
cl_int* positions;//[8] = {0,-1, -1,0, 1,0, 0,1}; 
cl_float* weights;//[4] = {1.0f,1.0f,1.0f,1.0f,}; 



SDKTimer *sampleTimer;
int timer;

cl_int eventStatus = CL_QUEUED;

cl_event ndrEvt;

CLCommandArgs   *ComandArgs;   /**< CLCommand argument class */

struct timeStruct{
	double kernelExecuting;
	double buildProgram;
	double setKernelArgs;
	double writeBack;
	double releaseKernel;
	double total;
	};

struct timeStruct times;

int convertToString(const char *filename, std::string& s);

void freeResources();

void StupidCPUimplementation(float *in, float *out, int width, int height);

void StupidDynamicCPUImplementation(float *in, float *out, int width, int height, 
					cl_int *positions, cl_float *allWeights, 
					cl_int numberPoints, cl_int edgewith);

int getPlatforms(void);

int getDevice(void);

int PrintDeviceInfo(int type);

int runCpuImplementation();

int buildProgram(cl_program *program);

int readArgs(int argc, char* argv[]);

//void getKernelArgSetError(int status);

int checkAgainstCpuImplementation(float *origInput, float *clOutput);

//void getExecutionError(int status);

int chekMemSimilar(float* openCl, float* referance, int length);

cl_int parseStringToPositions(std::string str);

cl_int parseStringToWeights(std::string str);

int setupKernelSpesificStuff(cl_uint* work_dim, size_t *global_work_size, size_t **local_work_size, 
				cl_context* context, cl_kernel* kernel, cl_kernel* kernelBackwards, cl_program* program);

//int createKernels(cl_kernel* kernel, cl_kernel* kernelBackwards, cl_program* program);

//int setWorkSizes(cl_uint* work_dim, size_t *global_work_size, size_t **local_work_size, 
//				cl_context* context, cl_kernel* kernel, cl_kernel* kernelBackwards);

cl_int getEdgeWidth();

void setInputEdgesToOne(cl_int edgewith);

void initilizeHostBuffers();

int setBufferKernelArgs(cl_kernel *kernel, cl_kernel *kernelBackwards, cl_context *context);

int runKernels(cl_kernel* kernel, cl_kernel* kernelBackwards, cl_command_queue* commandQueue, 
				size_t work_dim, size_t *global_work_size, size_t *local_work_size);

void printStats();