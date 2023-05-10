#!/usr/bin/bash
num_cpu=8
root_dir=`pwd`
log_dir=$root_dir/logs
rm -rf $log_dir
# result_dir=$root_dir/results
# result_dir=$result_dir/mnistdeeppoly.csv
# export RESULT_FILE=$result_dir
python script.py $num_cpu $log_dir
chmod -R u+x $log_dir/

# for((i=0; i<$num_cpu; i++))
# do
#     $log_dir"/script_$i.sh" &
# done