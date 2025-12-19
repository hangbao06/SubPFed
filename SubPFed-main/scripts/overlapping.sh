
python3 main.py --gpu $1\
                --n-workers $2\
                --model 'subpfed'\
                --dataset 'Cora' \
                --mode 'overlapping' \
                --frac 1.0 \
                --n-rnds 100\
                --n-eps 1\
                --n-clients 15\
                --norm-scale 5\
                --seed 42
