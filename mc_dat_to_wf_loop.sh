#!/bin/bash
COUNT=0
for I in {20..150..10}; do
    for J in {0..100}; do
        python mc_data_to_wf.py -i "/store/hep/users/yewzzang/JSNS2/rat_mc_positron/positron_${I}MeV/positron_${I}MeV_${J}.root" --energy ${I} --num ${J} &
        if [ $COUNT = 50 ]; then
            COUNT=0
            wait
        else
            COUNT=`expr $COUNT + 1`
        fi
    done
done
wait