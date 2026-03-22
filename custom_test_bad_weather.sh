export PYTHONPATH=$PYTHONPATH:./
export CUDA_VISIBLE_DEVICES=0

work_dir=work_dirs/${1}
config=${work_dir}/${1}.py
checkpoint=${work_dir}/${2}.pth
out=${work_dir}/results_${2}.pkl

echo "work directory: "${work_dir}
echo "config file: "${config}
echo "checkpoint: "${checkpoint}
echo "result file: "${out}

if [ -f "${out}" ]; then
    echo "Result file exists, using --result_file instead of --out"
    python ./tools/test.py \
        ${config} \
        ${checkpoint} \
        --result_file ${out} \
        --eval bbox \
        --eval-options bad_conditions=True \
        ${@:3}
else
    echo "Result file does not exist, using --out"
    python ./tools/test.py \
        ${config} \
        ${checkpoint} \
        --out ${out} \
        --eval bbox \
        --eval-options bad_conditions=True \
        ${@:3}
fi

