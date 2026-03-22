export PYTHONPATH=$PYTHONPATH:./
export CUDA_VISIBLE_DEVICES=0

config=projects/configs/${1}.py

echo "config file: "${config}

python ./tools/analysis_tools/get_flops.py \
        ${config} \
        ${@:2}
