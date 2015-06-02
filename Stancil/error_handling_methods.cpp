#include "CLPerfMarker.h"

using namespace std;

void buildProgramError(int status){

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

void getAppProfilerInitError(int status){
	
	switch (status){
	case AP_FINALIZED_PERF_MARKER:
		cout << " Finalized Perf Makrer:	clFinalizePerfMarker has already been called." << endl;
	break;
 	case AP_APP_PROFILER_NOT_DETECTED:
		cout << " App Profiler Not Detected:	The application has not been run with the AMD APP Profiler." << endl;
	break;
	}

}