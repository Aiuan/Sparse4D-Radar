export PYTHONPATH=$PYTHONPATH:./
export CUDA_VISIBLE_DEVICES=0

work_dir=work_dirs/${1}
config=${work_dir}/${1}.py
checkpoint=${work_dir}/${2}.pth

echo "work directory: "${work_dir}
echo "config file: "${config}
echo "checkpoint: "${checkpoint}

python ./tools/analysis_tools/benchmark.py \
        ${config} \
        ${checkpoint} \
        ${@:3}
