srun -C cpunode python run_cocabo_exps.py --func XGBFashionMNIST --max_itr 100 --kernel_mix 0.0 --seed;
srun -C cpunode python run_cocabo_exps.py --func XGBFashionMNIST --max_itr 100 --kernel_mix 0.5 --seed;
srun -C cpunode python run_cocabo_exps.py --func XGBFashionMNIST --max_itr 100 --kernel_mix 1.0 --seed;
