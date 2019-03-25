#!/bin/bash

echo '====================================='
echo "run of $2 with file $1 with assets $3"
echo '====================================='

cp "$1" ./simulate/battery_controller.py
if [ $# -eq 3 ]
then
    cp -r "$3" ./simulate/assets
fi
python ./simulate/simulate.py
rm -f ./simulate/battery_controller.py
rm -rf ./simulate/assets/

if [ -f "output/results.csv" ]
then
    echo "Script completed its run."
    echo ""
    cat ./output/results.csv
    echo ""
    mv ./output/results.csv ./all_results/"$2"_results.csv
else
    echo "ERROR: Script did execute correctly or timed out."
fi
