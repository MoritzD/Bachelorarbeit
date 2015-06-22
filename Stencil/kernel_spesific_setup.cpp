#include "kernel_spesific_setup.hpp"

using namespace std;

int setupKernelSpesificStuff(cl_uint* work_dim, size_t *global_work_size, size_t **local_work_size, 
				cl_context* context, cl_kernel* kernel, cl_kernel* kernelBackwards, cl_program* program){

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
			/* Create local mem objects to cash blocks in */
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

			*work_dim = 2;	
			global_work_size[0] = (width - 2);
			global_work_size[1] = 4;
			(*local_work_size)[0] = min((cl_uint)64,(width-2));
			(*local_work_size)[1] = 4;
			if(!ComandArgs->quiet){
				cout <<" height  and    width     "<< height << " " << width << endl;

				cout <<" global work size:  we ; he   "<< global_work_size[0] <<" ; "<< global_work_size[1] << endl;
	 
				cout <<" lokal work size:  we ; he   "<< (*local_work_size)[0] <<" ; " << (*local_work_size)[1] << endl;
			}
		break;
		case 5:
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
			
			if(StencilWeights.compare("default") == 0){
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
			cl_int edgewithlocal = getEdgeWidth();
			setInputEdgesToOne(edgewithlocal);
			edgewith = edgewithlocal;

			status = setBufferKernelArgs(kernel, kernelBackwards, context);
			if(status != SUCCESS){
				cout << "Abording!" << endl;
				freeResources();
				return FAILURE;
			}

			*work_dim = 2;
			global_work_size[0] = width - 2*edgewith;
			global_work_size[1] = height - 2*edgewith;
			//free(local_work_size);
			*local_work_size = NULL;//min((cl_uint)64,(width-2));

			

			cl_mem BufferPositions = clCreateBuffer(
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

			cl_mem BufferWeights = clCreateBuffer(
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
			status = clSetKernelArg(*kernel, 7, sizeof(cl_int), (void *)&edgewith);

			status = clSetKernelArg(*kernelBackwards, 4, sizeof(cl_mem), (void *)&BufferPositions);
			status = clSetKernelArg(*kernelBackwards, 5, sizeof(cl_mem), (void *)&BufferWeights);
			status = clSetKernelArg(*kernelBackwards, 6, sizeof(cl_int), (void *)&numberPoints);
			status = clSetKernelArg(*kernelBackwards, 7, sizeof(cl_int), (void *)&edgewith);
			
			if(VERBOSE){
				cout <<" working dimension: " << *work_dim << endl;
				cout <<" height  and    width     "<< height << " " << width << endl;

				cout <<" global work size:  we; he "<< global_work_size[0]<< global_work_size[1] << endl;
	 
				cout <<" lokal work size:  we "<< *local_work_size << endl;
			}
		break;
	}
	return SUCCESS;

}