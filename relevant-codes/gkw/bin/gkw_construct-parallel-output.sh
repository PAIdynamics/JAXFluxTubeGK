#!/bin/sh
# rebuild `parallel.dat' from par?????? using tail|head
# $Id: gkw_construct-parallel-output.sh 1005 2009-07-02 16:12:03Z phsgbq $
# N.B. assumes that the file number relates to the processor
# position in the s-direction

# check if we have any files!
if [ ! -s "par000000" ]
then
	echo "The file par000000 is missing or empty!"
    exit 1
fi

output_file="parallel.dat"
ntot=`grep NTOT par000000 | perl -pe 's/.*$\=\s*//g'`
ns=`grep NS par000000 | perl -pe 's/.*$\=\s*//g'`
file_list=`ls par[0-9][0-9][0-9][0-9][0-9][0-9] | perl -pe 's/par/\npar/g' | sort`

#check if the output file exists
if [ -e "$output_file" ]
then
	echo "The file $output_file exists - please (re)move."
	exit 1
fi

#write some basic things to the screen
echo "Input files:" $file_list
echo "NTOT = $ntot, NS = $ns"

# loop over slices and processors
for j in `seq $ntot -1 1` ; do
  k=$(echo $j*$ns | bc -l)
  for file in $file_list ; do
    tail -$k $file | head -$ns >> $output_file
  done
done
echo "Output file: $output_file"
