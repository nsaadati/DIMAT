# DIMAT
DIMAT: Decentralized Iterative Merging-And-Training for Deep Learning Models

## Install
Clone repo and install environment.yml and requirements.txt
~~~
conda env create --name DIMAT --file=environment.yml
pip install -r requirements.txt  # install
~~~

## Running training experiments
Example run for 5 ResNet-20 models on IID Cifar100: 
~~~
python -m train.train --dataset=cifar100 --model=resnet20 --num_models=5 --data_dist=iid --random_seed=0 > runs/train/output_cifar100_resnet20_num_models_5_iid_0.txt
~~~

Example of using train.sh shell script to automate experiments for varying number of agents and random seeds: 
~~~
./train.sh --nums_models "5 10" --data_dists "iid" --datasets_nums_classes "cifar100,100" --models "resnet20" --random_seeds "0 1 2 3 4" --num_parallel 10 --run_python true
~~~

## Running DIMAT experiments
Example run for DIMAT with 5 models using ResNet-20 on IID Cifar100: 
~~~
python -u -m main --dataset=cifar100 --wsize=5 --ckp=checkpoint/cifar100/resnet20/diff_initialization/models_no5/data_dist_iid/num_epochs_100/random_seed_0 --opt=DIMAT --merg_itr=30 --training --num_epochs=2 --randominit --seed=0 > runs/main/main_cifar100_resnet20_num_models_5_iid_DIMAT_30_num_epochs_2_0.txt
~~~

Example of using main.sh shell script to automate experiments for varying number of agents and random seeds: 
~~~
./main.sh --nums_models "5 10" --data_dists "iid" --datasets_nums_classes "cifar100,100" --models "resnet20" --random_seeds "0 1 2 3 4" --num_epochs "2" --opts "DIMAT WA" --num_parallel 10 --run_python true
~~~

## Running baseline experiments
Example run for CDSGD with 5 models using ResNet-20 on IID Cifar100: 
~~~
python -m torch.distributed.launch --master_port 0 --nnodes 1 --nproc_per_node=5 base_lines.py --data=cifar100 --model=resnet20 --ckp=checkpoint/cifar100/resnet20/diff_initialization/models_no5/data_dist_non-iid/num_epochs_100/random_seed_0 --opt=CDSGD --epochs=200 --randominit=False --diffinit=False --num_epochs=2 --data_dist=non-iid --exp=1 --seed=0 > runs/base_lines/base_lines_cifar100_resnet20_num_models_5_non-iid_CDSGD_200_num_epochs_2_randominitFalse_diffinitFalse_exp_1_0.txt
~~~

Example of using base_lines.sh shell script to automate experiments for varying baselines and random seeds: 
~~~
./base_lines.sh --nums_models "5" --exps "1 2"  --data_dists "non-iid iid" --datasets_nums_classes "cifar100,100" --models "resnet20" --random_seeds "0 1 2 3 4" --nums_epochs "2" --opts "CDSGD SGP" --num_parallel 1 --randominits False --diffinits False --run_python true
~~~

## List of Datasets (-dataset argument)
- CIFAR10 (cifar10)
- CIFAR100 (cifar100)
- Tiny ImgNet (tinyimgnt)

## Model Architecture (-model argument)
- ResNet-20 (resnet20)
- ResNet-50 (resnet50)
- VGG16 (vgg16)

## Data Distributions (-data_dist argument)
- IID (iid)
- non-IID (non-iid)

## Algorithm (-opt argument) - For main.py
- DIMAT (DIMAT)
- WA (WA)
- WM (WM)

## Algorithm (-opt argument) - For base_lines.py
- SGP (SGP)
- CGA (CGA)
- CDSGD (CDSGD)

## Acknowledgements


## Citation
Please cite our paper in your publications if it helps your resemodel:

	@article{article,
	  title={title},
	  author={author},
	  journal={journal},
	  year={2022}
	}

## Paper Links
[DIMAT: Decentralized Iterative Merging-And-Training
for Deep Learning Models](paper_link)


## Contributors
- []()
