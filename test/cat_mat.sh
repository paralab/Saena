#!/bin/bash
# concatenate the parallel matrix and rhs output files of Saena to single files
v="cat"
v2="cat"
for i in {0..279}
do
#   echo "i = $i"
   v="$v mat-r$i.mtx"
   v2="$v2 rhs-r$i.txt"
done
v="$v > mat.mtx"
v2="$v2 > rhs.txt"
echo $v
eval $v
echo $v2
eval $v2

