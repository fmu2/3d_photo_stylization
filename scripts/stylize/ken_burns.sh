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

    python batch_stylize.py \
        -m log/$1/stylize.pth \
        -n $1_$i \
        -ldi test/content/ldi/eval/$i.mat \
        -s test/style \
        -ns 130 \
        -pc 2 \
        -f 90 \
        -cam ken_burns \
        -x "$x1" "$x2" \
        -y "$y1" "$y2" \
        -z 0 "$z" \
        -g $2

    echo "$i done"
done