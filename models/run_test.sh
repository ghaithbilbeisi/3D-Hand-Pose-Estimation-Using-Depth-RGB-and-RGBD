#!/bin/bash

model=$1
caffemodel=$2
iterations=$3
output=$4
dataset=$5
n=$((-1*((55056/iterations)*63*2+1)))
clean_out=$(echo ${output/.txt/_.txt})

caffe test -model $model -weights $caffemodel -phase TEST -iterations $iterations -gpu 0 2>&1 | tee ../logs/$output
cd ../logs
head -n $n $4 | tee $clean_out
sed '/labels =/!d' ./$clean_out > $(echo ${clean_out/_.txt/_label.txt})
sed '/predict =/!d' ./$clean_out > $(echo ${clean_out/_.txt/_predict.txt})
cd ../evaluation\ pretty ### change to ../evaluation after everything is done
python3 compute_error.py $dataset max-frame $clean_out
python3 compute_error.py $dataset mean-frame $clean_out
python3 compute_error.py $dataset joint $clean_out
