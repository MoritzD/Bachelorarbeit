# Stancil 
## calculations for OpenCL

### Required software:

1. a OpenCL implementation
2. cmake and make

### How to Compile:
from the base directroy of this project:

1. create the Makefile using cmake: run cmake -G "Unix Makefiles" Stancil/
2. Built the project by runing: make

### Example Call:

./Stancil -we 4034 -he 4034 -i 100 -kv 2
	using Matrix of size 4034 x 4034 
	run 100 iterrations
	using kernel 2


### Usage:

-h, --help   
			
			Prints out a help text

-q, --quiet
			
			Quiet mode. Suppress all additional text output.

-e, --verify
			
			Don't Verify results against reference implementation.

-we, --width
			
			Width of matrix

-he, --height
			
			height of matrix

-i, --iterations
			
			number of iterations

-kv, --kernelversion
			
			witch version of the kernel shold be used
			accepted values:
			1,2,3,4,5,6
			where: 1-5: 5-Point Stancil
			2 is Fastest Standard implementation
			5 is not working at the moment
			6: Dynamic Stancil

-st, --stancil
			
			define what kind of stanil should be used (only needed for kernel 6; will be discarded otherwise)
			e.g.: -st "-1,0, 0,-1, 0,1, 1,0" for the 5-Point-Stancil
			every 2 numbers are grouped together as a (x,y)-coordinate relative to the center point.
			expects a String with the numbers seperated by ','

			if not specifyed and kernel 6 is used the default is 5-Point-Stancil

-w, --weights
			
			define weights for every point in the dynamic stancil
			e.g. -w "1.0, -2.4, 12.34, 0.5"
			a String with the weights for the points. Seperatet by ','

			if not specifyed and kernel 6 is used all weights will be set to 1.0

-d, --device
			
			define what kind of device should be used
			accepted values:
			[gpu | cpu | all | stupid]
			all: uses gpu if available cpu otherwise
			stupid: uses the "stupid cpu implementation" so the cpu referance

-V, --verbose
			
			gives more output

-VM, --verbosematirx
			
			print out calculated matrix

-VK, --verbosekernel
			
			print out kernel file