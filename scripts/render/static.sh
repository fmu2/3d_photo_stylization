#!/bin/bash

for i in $(cat test/eval_ids.txt)
do
    python test_ldi_render.py \
    -n $i \
    -ldi test/content/ldi/eval/$i.mat \
    -aa 2 \
    -f 1 \
    -cam static \
    -g $1

    echo "$i done"
done
