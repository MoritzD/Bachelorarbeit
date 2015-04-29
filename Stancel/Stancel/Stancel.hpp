
#include <CL/cl.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <fstream>
#include <ctime>
#include "CLUtil.hpp"


using namespace appsdk;



#define SUCCESS 0
#define FAILURE 1 
#define END 99


#define VERBOSE false
#define SINGLETIME true 

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
size_t valueSize;
char* value;
cl_uint memsize, j;
cl_uint maxComputeUnits;
cl_uint DeviceToUse = 0;
cl_uint width = WIDTH, height = HEIGHT;
cl_uint iterations = ITERATIONS;
cl_uint kernelVersion = 3;

cl_float *input = NULL;
cl_float *output = NULL;

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

void freeResources();

void StupidCPUimplementation(float *in, float *out, int width, int height);

int getPlatforms(void);

int getDevice(void);

int PrintDeviceInfo(int type);

int runCpuImplementation();

int buildProgram(cl_program *program);

int readArgs(int argc, char* argv[]);

void getKernelArgSetError(int status);

int checkAgainstCpuImplementation(float *origInput, float *clOutput);

void getExecutionError(int status);
