#!/bin/bash

# If you don't have permission to execute the shell script use the following line in the terminal
# chmod +x train.sh

# Terminate child processes of the current script's process ID if the shell script itself is terminated
function kill_background_processes() {
    pkill -P $$
}
trap "kill_background_processes; exit 1" SIGINT SIGTERM

# Create the 'runs' folder if it does not exist
if [ ! -d "runs/train" ]; then
    mkdir -p runs/train
fi

# Only run Python scripts when specifified, add " --run_python true " to the command after checking the printed experiment list
run_python=false

# =====================================================================================================================================================================
# Arguments - Modify for desired experiments or use command-line arugments! 
# Commands should be of the following format:
# ./train.sh --nums_models "2 5 10 20" --data_dists "iid non-iid" --datasets_nums_classes "cifar100,100" --models "resnet20" --random_seeds "0 1 2 3 4" --num_parallel 2
# =====================================================================================================================================================================
# All combinations
nums_models=(2 5 10 20)
data_dists=("iid" "non-iid" "non-iid-PF")
datasets_nums_classes=("cifar100,100" "tinyimgnt,200" "cifar10,10")
models=('resnet20' 'resnet50' 'vgg16')
random_seeds=(0 1 2 3 4)
epochs=(100)

# If hardware allows, run multiple scripts in parallel (1 does a single script at a time)
# Each instance of the script can use up to around 3 GB of GPU Memory (check GPU with nvidia-smi)
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
        --epochs)
            epochs=$2
            shift 2
            ;;
        --num_parallel)
            num_parallel=$2
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
echo "epochs: ${epochs[@]}"
echo "random_seed: ${random_seeds[@]}"
echo "num_parallel: $num_parallel"

# Loop over arguments
for dataset_num_classes in "${datasets_nums_classes[@]}"; do
  # Separate the dataset and num_classes pair
  IFS=',' read -r dataset num_classes <<< "$dataset_num_classes"

  for model in "${models[@]}"; do
    for num_models in "${nums_models[@]}"; do
      # Filter out values in nums_models that are not a factor of (but not equal to) the number of classes
      #if [ $((num_classes % num_models)) -eq 0 ] && [ "$num_classes" -ne "$num_models" ]; then 
        for data_dist in "${data_dists[@]}"; do
          for random_seed in "${random_seeds[@]}"; do
            # Save console otuputs to text files since they will not be displayed otherwise - can check progress by opening the file at the printed path
            output_filename="runs/train/train_${dataset}_${model}_num_models_${num_models}_${data_dist}_${random_seed}.txt"

            # Print current command
            echo "python -u -m train --dataset=${dataset} --model=${model} --num_models=${num_models} --data_dist=${data_dist} --random_seed=${random_seed} --epochs=${epochs} > ${output_filename}"

            # Add " --run_python true " to the command after checking the printed experiment list
            if [ "$run_python" = true ]; then
              # Run the Python script with the specified arguments
              python -u -m train --dataset="$dataset" --model="$model" --num_models="$num_models" --data_dist="$data_dist" --seed="$random_seed" --epochs="$epochs" 2>&1 | tee "$output_filename" &
            fi

            # Increment the parallel scripts counter
            ((i++))

            # Check if the maximum number of concurrent scripts has been reached
            if [ $i -ge $num_parallel ] && [ "$run_python" = true ]; then
              # Wait for the background processes to finish
              echo "Wait for the background processes to finish, check progress with the text files"
              wait

              # Reset the counter
              i=0
            fi
          done
        done
      #fi
    done
  done
done

echo "Done"