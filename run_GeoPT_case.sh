#!/bin/bash
# Run MLEvolve on car_crash prediction task (HDF5 data)
# Usage: bash run_car.sh

# 使用阿里云镜像
export HF_ENDPOINT=https://hf-mirror.com

#cd /home/cjl/mlevolve-code/MLEvolve

# Generate timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="runs/car_run_${TIMESTAMP}.log"

# Ensure runs directory exists
mkdir -p runs

# Run in background
nohup python run.py \
  exp_id="GeoPT_case1" \
  goal="" \
  data_dir="" \
  exp_name="GeoPT_case1-h5-prediction-with-globalmem" \
  dataset_dir="" \
  preprocess_data=False \
  > "${LOG_FILE}" 2>&1 &

echo "carcrash prediction task started in background."
echo "Log file: ${LOG_FILE}"
echo "Monitor with: tail -f ${LOG_FILE}"
echo ""
echo "Stop the task with: pkill -f 'python run.py'"
echo "Or find and kill the process: ps aux | grep run.py && kill <PID>"
