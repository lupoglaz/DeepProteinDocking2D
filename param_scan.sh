#Scanning step_size in Langevin dynamics
# python train_docking.py -experiment ParamScan_stepsize_0.1 -step_size 0.1
# python results.py -experiment ParamScan_stepsize_0.1

# python train_docking.py -experiment ParamScan_stepsize_0.5 -step_size 0.5
# python results.py -experiment ParamScan_stepsize_0.5

python train_docking.py -experiment ParamScan_stepsize_5.0 -step_size 5.0
python results.py -experiment ParamScan_stepsize_5.0

python train_docking.py -experiment ParamScan_stepsize_10.0 -step_size 10.0
python results.py -experiment ParamScan_stepsize_10.0

python train_docking.py -experiment ParamScan_stepsize_25.0 -step_size 25.0
python results.py -experiment ParamScan_stepsize_25.0