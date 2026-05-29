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
  exp_id="car_crash" \
  goal="Car-crash simulation is a core task of industrial-scale solid mechanics simulation for assessing vehicle structural safety under high-speed impact. The car-crash benchmark derives from the Neon full-vehicle model of the National Crash Analysis Center (NCAC), a widely adopted industrial standard crash benchmark equipped with detailed part-level meshing and heterogeneous material property assignment. This simulation targets predicting the maximum 2D Von Mises equivalent stress of each finite element during the whole dynamic collision process.Collision angle is set as the sole variable and uniformly sampled within [-45°, 45°]; even with fixed initial geometry, different angles produce completely different contact sequences, load transfer paths and structural deformation modes." \
  data_dir="/data2/xinyi/carcrash.h5" \
  exp_name="carcrash-h5-prediction-with-globalmem" \
  dataset_dir="/data2/xinyi" \
  preprocess_data=False \
  > "${LOG_FILE}" 2>&1 &

echo "carcrash prediction task started in background."
echo "Log file: ${LOG_FILE}"
echo "Monitor with: tail -f ${LOG_FILE}"
echo ""
echo "Stop the task with: pkill -f 'python run.py'"
echo "Or find and kill the process: ps aux | grep run.py && kill <PID>"
