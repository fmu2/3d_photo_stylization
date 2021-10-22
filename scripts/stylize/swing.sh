#!/bin/bash

x=0.02
z=0.05

for i in $(cat test/eval_ids.txt)
do
    n=$(( $RANDOM % 2 ))
    if [[ "$n" -eq 0 ]]; then x1=-$x; x2=$x
    else x1=$x; x2=-$x
    fi

    python batch_stylize.py \
        -m log/$1/stylize.pth \
        -n $1_$i \
        -ldi test/content/ldi/eval/$i.mat \
        -s test/style \
        -ns 130 \
        -ndc \
        -pc 2 \
        -f 90 \
        -cam swing \
        -x "$x1" "$x2" \
        -y 0 0 \
        -z 0 "$z" \
        -g $2

    echo "$i done"
done