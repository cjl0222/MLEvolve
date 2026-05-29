#!/bin/bash
# Run MLEvolve on fire prediction task (HDF5 data)
# Usage: bash run_fire.sh

# 使用阿里云镜像
export HF_ENDPOINT=https://hf-mirror.com

#cd /home/cjl/mlevolve-code/MLEvolve

# Generate timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="runs/fire_run_${TIMESTAMP}.log"

# Ensure runs directory exists
mkdir -p runs

# Run in background
nohup python run.py \
  exp_id="cylindrical_charge" \
  goal="physical simulation of fluid mechanics (cars/aircraft/ships),tabular regression tasks" \
  data_dir="/data2/xinyi/fire.h5" \
  exp_name="fire-h5-prediction-with-globalmem" \
  dataset_dir="/data2/xinyi" \
  preprocess_data=False \
  > "${LOG_FILE}" 2>&1 &

echo "Fire prediction task started in background."
echo "Log file: ${LOG_FILE}"
echo "Monitor with: tail -f ${LOG_FILE}"
echo ""
echo "Stop the task with: pkill -f 'python run.py'"
echo "Or find and kill the process: ps aux | grep run.py && kill <PID>"
