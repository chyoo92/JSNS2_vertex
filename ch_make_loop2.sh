#!/bin/bash
COUNT=0
for J in `seq 100 4868`; do
    python ch_1563_make.py --runnum ${J} &
    if [ $COUNT = 50 ]; then
        COUNT=0
        wait
    else
        COUNT=`expr $COUNT + 1`
    fi
done
wait