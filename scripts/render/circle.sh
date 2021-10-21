#!/bin/bash

x=0.02
y=0.02
z=0.05

for i in $(cat test/eval_ids.txt)
do
    n=$(( $RANDOM % 4 ))
    if [[ "$n" -eq 0 ]]; then x1=-$x; x2=$x; y1=-$y; y2=$y
    elif [[ "$n" -eq 1 ]]; then x1=$x; x2=-$x; y1=-$y; y2=$y
    elif [[ "$n" -eq 2 ]]; then x1=-$x; x2=$x; y1=$y; y2=-$y
    else x1=$x; x2=-$x; y1=$y; y2=-$y
    fi

    python test_ldi_render.py \
    -n $i \
    -ldi test/content/ldi/eval/$i.mat \
    -aa 2 \
    -f 90 \
    -cam circle \
    -x "$x1" "$x2" \
    -y "$y1" "$y2" \
    -z 0 "$z" \
    -g $1
done