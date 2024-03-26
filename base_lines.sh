#!/bin/bash

# If you don't have permission to execute the shell script use the following line in the terminal
# chmod +x base_lines.sh

# Terminate background processes if the shell script itself is terminated
function kill_background_processes() {
    pkill -P $$  # Kill all child processes of the current script's process ID
}
trap "kill_background_processes; exit 1" SIGINT SIGTERM

# Create the 'runs' folder if it does not exist
if [ ! -d "runs/base_lines" ]; then
    mkdir -p runs/base_lines
fi

# Only run Python scripts when specifified, add " --run_python true " to the command after checking the printed experiment list
run_python=false
# =====================================================================================================================================================================
# Arguments - Modify for desired experiments or use command-line arugments! 
# Commands should be of the following format:
# ./main.sh --nums_models "2 5 10 20" --data_dists "iid non-iid" --datasets_nums_classes "cifar100,100" --models "resnet20" --random_seeds "0 1 2 3 4" --num_parallel 2
# =====================================================================================================================================================================
# All combinations
nums_models=(2 5 10 20)
data_dists=("iid" "non-iid" "non-iid-PF")
datasets_nums_classes=("cifar100,100" "tinyimgnt,200" "cifar10,10")
models=('resnet20' 'resnet50' 'vgg16')
randominits=('False')
diffinits=('False')
random_seeds=(0 1 2 3 4)
exps=(1)
# New arguments for main.py
num_epochs=(2, 5, 10, 20)
epoch=(200)
opts=("SGP" "CGA")

# randominit # manually add flag for one set of experiments later, for supplementary

# If hardware allows, run multiple scripts in parallel (1 does a single script at a time)
# Each instance of the script uses around 3 GB of GPU Memory (check GPU with nvidia-smi)
num_parallel=2

# Counter to make sure too many scripts aren't started
i=0

# Process command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --nums_models)
            nums_models=($2)
            shift 2
            ;;
        --data_dists)
            data_dists=($2)
            shift 2
            ;;
        --datasets_nums_classes)
            datasets_nums_classes=($2)
            shift 2
            ;;
        --models)
            models=($2)
            shift 2
            ;;
        --random_seeds)
            random_seeds=($2)
            shift 2
            ;;
        --num_parallel)
            num_parallel=$2
            shift 2
            ;;
        --nums_epochs)
            nums_epochs=($2)
            shift 2
            ;;
        --epoch)
            epoch=($2)
            shift 2
            ;;
        --randominits)
            randominits=($2)
            shift 2
            ;;
        --diffinits)
            diffinits=($2)
            shift 2
            ;;
        --opts)
            opts=($2)
            shift 2
            ;;
        --exps)
            exps=($2)
            shift 2
            ;;
        --run_python)
            run_python=$2
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Print processed input arguments
echo "Input Arguments:"
echo "num_models: ${nums_models[@]}"
echo "data_dist: ${data_dists[@]}"
echo "datasets_nums_classes: ${datasets_nums_classes[@]}"
echo "model: ${models[@]}"
echo "random_seed: ${random_seeds[@]}"
echo "num_epochs: ${nums_epochs[@]}"
echo "epoch: ${epoch[@]}"
echo "randominit: ${randominits[@]}"
echo "diffinit: ${diffinits[@]}"
echo "exps: ${exps[@]}"
echo "opt: ${opts[@]}"
echo "num_parallel: $num_parallel"

export CUDA_VISIBLE_DEVICES=0

# Loop over arguments
for dataset_num_classes in "${datasets_nums_classes[@]}"; do
  # Separate the dataset and num_classes pair
  IFS=',' read -r dataset num_classes <<< "$dataset_num_classes"

  for model in "${models[@]}"; do
    for num_models in "${nums_models[@]}"; do
      # Filter out values in nums_models that are not a factor of (but not equal to) the number of classes
      #if [ $((num_classes % num_models)) -eq 0 ] && [ "$num_classes" -ne "$num_models" ]; then 
        for data_dist in "${data_dists[@]}"; do
         for randominit in "${randominits[@]}"; do
         for diffinit in "${diffinits[@]}"; do
        # Main.py arugments
          for num_epochs in "${nums_epochs[@]}"; do
            for epochs in "${epoch[@]}"; do
             for exp in "${exps[@]}"; do
               for opt in "${opts[@]}"; do
                 for random_seed in "${random_seeds[@]}"; do
                  # Save console otuputs to text files since they will not be displayed otherwise - can check progress by opening the file at the printed path
                  output_filename="runs/base_lines/base_lines_${dataset}_${model}_num_models_${num_models}_${data_dist}_${opt}_${epochs}_num_epochs_${num_epochs}_randominit${randominit}_diffinit${diffinit}_exp_${exp}_${random_seed}.txt"

                  # Get checkpoint path given current arguments
                  checkpoint="checkpoint/$dataset/$model/diff_initialization/models_no$num_models/data_dist_$data_dist/num_epochs_100/random_seed_0"

                  # Print current command
                  echo "python -m torch.distributed.launch --master_port 0 --nnodes 1 --nproc_per_node=${num_models} base_lines.py --data=${dataset} --model=${model} --ckp=${checkpoint} --opt=${opt} --epochs=${epochs} --randominit=${randominit} --diffinit=${diffinit} --num_epochs=${num_epochs} --data_dist=${data_dist} --exp=${exp} --seed=${random_seed} > ${output_filename}"

                  # Add " --run_python true " to the command after checking the printed experiment list
                  if [ "$run_python" = true ]; then
                    # Run the Python script with the specified arguments
                    python -m torch.distributed.launch --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes 1 --nproc_per_node="$num_models" base_lines.py --data="$dataset" --model="$model" --ckp="$checkpoint" --opt="$opt" --epochs="$epochs" --randominit="$randominit" --diffinit="$diffinit" --num_epochs="$num_epochs"  --data_dist="$data_dist" --exp="$exp" --seed="$random_seed" 2>&1 | tee "$output_filename" &
                  fi

                  # Increment the parallel scripts counter
                  ((i++))

                  # Check if the maximum number of concurrent scripts has been reached
                  if [ $i -ge $num_parallel ] && [ "$run_python" = true ]; then
                    # Wait for the background processes to finish
                    echo "Wait for the background processes to finish, check progress with the text file outputs"
                    wait

                    # Reset the counter
                    i=0
                  fi
                done
               done
              done
             done
            done
          done
        done
       done
      #fi
    done
  done
done

echo "Done"