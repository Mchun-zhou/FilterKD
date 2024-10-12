PROJECT_PATH=
BASE_MODEL=
DATA_PATH=
DATASTORE_LOAD_PATH=
RESULTS_PATH= #保存输出记录的txt位置目录

K_VALUES=(4 8 16 32 64 128 256 512)
LAMBDA_VALUES=(0.2 0.3 0.4 0.5 0.6 0.7 0.8)
TEMPERATURE_VALUES=(10.0 100.0 200.0 500.0 1000.0)
echo "knn-k,knn-lambda,knn-temperature,bleu_score" > $RESULTS_PATH
for knn_k in "${K_VALUES[@]}"; do
  for knn_lambda in "${LAMBDA_VALUES[@]}"; do
    for knn_temperature in "${TEMPERATURE_VALUES[@]}"; do
      echo "Evaluating: knn-k=$knn_k, knn-lambda=$knn_lambda, knn-temperature=$knn_temperature"
      CUDA_VISIBLE_DEVICES=0 python $PROJECT_PATH/knnbox-scripts/common/generate.py $DATA_PATH \
      --task translation \
      --path $BASE_MODEL \
      --dataset-impl mmap \
      --beam 5 \
      --max-len-a 1.2 --max-len-b 10 \
      --source-lang de --target-lang en \
      --gen-subset valid \
      --max-tokens 30720 \
      --scoring sacrebleu \
      --tokenizer moses \
      --remove-bpe \
      --user-dir $PROJECT_PATH/knnbox/models \
      --arch vanilla_knn_mt@transformer_iwslt_de_en \
      --knn-mode inference \
      --knn-datastore-path $DATASTORE_LOAD_PATH \
      --knn-k $knn_k \
      --knn-lambda $knn_lambda \
      --knn-temperature $knn_temperature > temp_result_deen.txt
      # 提取BLEU分数，确保正确提取数值部分
      BLEU_SCORE=$(grep -oP "(?<=BLEU = )\d+\.\d+" temp_result_deen.txt)
      # 保存参数组合和对应的BLEU分数
      echo "$knn_k,$knn_lambda,$knn_temperature,$BLEU_SCORE" >> $RESULTS_PATH
    done
  done
done