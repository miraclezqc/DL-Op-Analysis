# DL-Op-Analysis

## Requirems
- Linux
- Python3 (with torch)
- CUDA (with ncu)
## Environment Setting
```
export PYTHONPATH=/your-path/DL-Op-Analysis/common:$PYTHONPATH
```

## Nsight Compute Command
```
metrics="sm__cycles_elapsed.avg,sm__cycles_elapsed.avg.per_second,\
sm__sass_thread_inst_executed_op_dadd_pred_on.sum,sm__sass_thread_inst_executed_op_dfma_pred_on.sum,sm__sass_thread_inst_executed_op_dmul_pred_on.sum,\
sm__sass_thread_inst_executed_op_fadd_pred_on.sum,sm__sass_thread_inst_executed_op_ffma_pred_on.sum,sm__sass_thread_inst_executed_op_fmul_pred_on.sum,\
sm__sass_thread_inst_executed_op_hadd_pred_on.sum,sm__sass_thread_inst_executed_op_hfma_pred_on.sum,sm__sass_thread_inst_executed_op_hmul_pred_on.sum,\
sm__inst_executed_pipe_tensor.sum,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,lts__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__time_duration.sum,\
dram__bytes.sum,lts__t_bytes.sum,l1tex__t_bytes.sum"

/yourpath/cuda/bin/ncu --metrics $metrics --csv --target-processes all python3 ./Loss/mse/profile.py > nsight.csv
```

