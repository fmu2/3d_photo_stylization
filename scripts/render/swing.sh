#!/bin/bash

x=0.02
z=0.05

x1=-$x; x2=$x

for i in $(cat test/eval_ids.txt)
do
    n=$(( $RANDOM % 2 ))
    if [[ "$n" -eq 0 ]]; then x1=-$x; x2=$x
    else x1=$x; x2=-$x
    fi

    python test_ldi_render.py \
    -n $i \
    -ldi test/content/ldi/eval/$i.mat \
    -aa 2 \
    -f 90 \
    -cam swing \
    -x "$x1" "$x2" \
    -y 0 0 \
    -z 0 "$z" \
    -g $1
done
