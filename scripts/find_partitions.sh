#!/bin/bash
# find_partitions.sh - Discover available SLURM partitions and GPU resources

echo "🔍 Discovering SLURM Configuration on Hellgate"
echo "=============================================="

echo "📋 Available Partitions:"
sinfo -o "%20P %5a %10l %6D %6t %14C %8z %15m %8G %N"

echo ""
echo "🎯 GPU-Enabled Partitions:"
sinfo -o "%20P %5a %10l %6D %6t %14C %8z %15m %8G %N" | grep -i gpu

echo ""
echo "📊 Detailed Partition Information:"
sinfo -Nel

echo ""
echo "🖥️  GPU Resources:"
sinfo -o "%20P %20G" | grep -v "gpu:0" | grep "gpu"

echo ""
echo "⚙️  Checking for specific GPU partitions:"
partitions=$(sinfo -h -o "%P" | tr -d '*' | sort -u)
echo "All partitions: $partitions"

for partition in $partitions; do
    gpu_info=$(sinfo -p $partition -o "%G" -h | head -1)
    if [[ "$gpu_info" != "(null)" && "$gpu_info" != "gpu:0" ]]; then
        echo "✅ Partition '$partition' has GPUs: $gpu_info"
    fi
done

echo ""
echo "🏗️  Sample job commands for different scenarios:"
echo ""

# Find the correct GPU partition
gpu_partitions=$(sinfo -h -o "%P %G" | grep -v "gpu:0" | grep "gpu" | awk '{print $1}' | tr -d '*' | sort -u)

if [ -n "$gpu_partitions" ]; then
    first_gpu_partition=$(echo "$gpu_partitions" | head -1)
    echo "🚀 Recommended SLURM commands:"
    echo ""
    echo "1. Interactive GPU session:"
    echo "   srun -p $first_gpu_partition --gres=gpu:1 --time=01:00:00 --pty bash"
    echo ""
    echo "2. Submit GPU test job:"
    echo "   # Edit partition in gpu_test_job.sh to: $first_gpu_partition"
    echo "   sbatch scripts/gpu_test_job.sh"
    echo ""
    echo "3. Submit full training job:"
    echo "   # Edit partition in slurm_gpu_fixed.sh to: $first_gpu_partition"
    echo "   sbatch scripts/slurm_gpu_fixed.sh"
    
else
    echo "⚠️  No obvious GPU partitions found. Trying common names:"
    common_names=("gpu" "gpu-v100" "gpu-a100" "gpu-rtx" "compute" "general" "batch")
    
    for name in "${common_names[@]}"; do
        if sinfo -p "$name" &>/dev/null; then
            echo "✅ Partition '$name' exists"
        fi
    done
    
    echo ""
    echo "🔍 Manual check required. Try:"
    echo "   sinfo -Nel | grep -i gpu"
    echo "   scontrol show partition"
fi

echo ""
echo "📞 Need help? Contact your system administrator or check:"
echo "   - Cluster documentation"
echo "   - Run: scontrol show partition"
echo "   - Run: sacctmgr show associations"
