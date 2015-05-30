#! /bin/bash
if [  $# -eq 0 ]
	then
		echo "Help:"
		echo "Script expects the filename it shuld write to as a parameter"
		echo "Script only executes, without varifying the results"
		echo "Example: ./benchmark.sh outputfile.txt"
		exit 0
fi

echo "Benchmarking script; writing to file: "$1


if [ ! -e Stancil/bin/x86_64/Release/Stancil ]
	then
		echo "Executable not found, try to compile it"
		make
		if [ ! -e Stancil/bin/x86_64/Release/Stancil ]
			then
				echo "compiling faild still can't find the Executable; Abording"
				exit 0
		fi
fi

if [  $# -eq 2 ]
	then
		if [ $2 == new ]
			then
				echo "overriting file if existing"
				echo "Benchmark results from: "	> $1
				date >> $1

			else
				echo "Adding to file if existing"
				echo "Benchmark results from: "	>> $1
				date >> $1
		fi
	else
		echo "Adding to file if existing"
		echo "Benchmark results from: "	>> $1
		date >> $1
fi

for i in `seq 1 6`
do
	echo "Calculating kernel "$i
	echo "" >> $1
	echo "" >> $1
	echo "Kernel "$i":" >> $1
	./Stancil/bin/x86_64/Release/Stancil -we 4034 -he 4034 -i 100 -e -kv $i -q >> $1
done

exit 0