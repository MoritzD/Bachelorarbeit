#include "Stencil.cpp"

using namespace std;

int setupKernelSpesificStuff(cl_uint* work_dim, size_t *global_work_size, size_t **local_work_size, 
				cl_context* context, cl_kernel* kernel, cl_kernel* kernelBackwards, cl_program* program);