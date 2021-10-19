#!/bin/bash

x=0.02
y=0.02
z=0.05

x1=-$x; x2=$x
y1=-$y; y2=$y
z1=0; z2=$z

for i in $(cat test/eval_ids.txt)
do
    python batch_stylize.py \
        -m log/$1/stylize.pth \
        -n $1_$i \
        -ldi test/content/ldi/eval/$i.mat \
        -s test/style \
        -ns 130 \
        -pc 2 \
        -f 90 \
        -cam circle \
        -x "$x1" "$x2" \
        -y "$y1" "$y2" \
        -z "$z1" "$z2" \
        -g $2
done