#!/bin/bash
# rename files
v="rhs_piece_ref2_mode4-r"
v2="rhs-r"
for i in {0..279}
do
#   echo "i = $i"
   mv "$v$i.txt" "$v2$i.txt"
done
