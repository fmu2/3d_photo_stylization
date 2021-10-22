#!/bin/bash

for i in $(cat test/eval_ids.txt)
do
    python batch_stylize.py \
    	-m log/$1/stylize.pth \
    	-n $1_$i \
    	-ldi test/content/ldi/eval/$i.mat \
    	-s test/style \
    	-ns 130 \
        -ndc \
    	-pc 2 \
    	-f 1 \
    	-cam static \
    	-g $2

    echo "$i done"
done