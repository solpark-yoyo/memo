#!/usr/bin/env bash
set -euo pipefail

sd_versions=(1 2)       # 1 2 99 (For Realistic Vision)
gen_numbers=(1)
guidance_scales=(7.5)
mode=("x,c|x")
seeds=(51)
normalization=("None")  # "L1" "L2" "None"

export CUDA_VISIBLE_DEVICES=0

for normalization_type in "${normalization[@]}"; do
    for seed in "${seeds[@]}"; do
        for sd_ver in "${sd_versions[@]}"; do
            if [[ "$sd_ver" == "1" ]]; then
                data_paths=('prompts/sd1_mem.txt' 'prompts/sd1_nmem.txt')
            elif [[ "$sd_ver" == "2" ]]; then
                data_paths=('prompts/sd2_mem.txt' 'prompts/sd2_nmem.txt')
            elif [[ "$sd_ver" == "99" ]]; then
                data_paths=('prompts/RV_mem.txt' 'prompts/RV_nmem.txt')
            else
                echo "Invalid sd_ver: $sd_ver"; continue
            fi

            for mode_type in "${mode[@]}"; do
                for guidance_scale in "${guidance_scales[@]}"; do
                    for data_path in "${data_paths[@]}"; do
                        for gen_num in "${gen_numbers[@]}"; do
                            echo "sd_ver=$sd_ver data_path=$data_path gen_num=$gen_num mode=$mode_type"
                            python detect_mem.py \
                                --sd_ver "$sd_ver" \
                                --data_path "$data_path" \
                                --gen_num "$gen_num" \
                                --guidance_scale "$guidance_scale" \
                                --mode "$mode_type" \
                                --gen_seed "$seed" \
                                --normalization "$normalization_type"
                        done
                    done
                done
            done
        done
    done
done

echo "All combinations done."
