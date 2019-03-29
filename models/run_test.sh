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
cd ../evaluation
python3 compute_error.py REN_9x6x6 max-frame $dataset $clean_out
python3 compute_error.py REN_9x6x6 mean-frame $dataset $clean_out
python3 compute_error.py REN_9x6x6 joint $dataset $clean_out
