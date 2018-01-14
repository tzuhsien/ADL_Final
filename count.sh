#!/bin/sh

for i in $(find records/ -name "12*" -exec basename {} \;); do 
    echo $i | cut -d _ -f 2; 
done | awk '{s+=$1} END {print s}'
