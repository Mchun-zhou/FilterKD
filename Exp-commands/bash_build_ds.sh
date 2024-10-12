PROJECT_PATH=
BASE_MODEL= #base模型的位置，低资源单独训练transformer_iwslt_de_en ;领域翻译使用wmt19预训练模型
DATASET= #标识
DATA_PATH= #数据位置
DATASTORE_SAVE_PATH= #数据存储保存位置
CUDA_VISIBLE_DEVICES=0 python $PROJECT_PATH/knnbox-scripts/common/validate.py $DATA_PATH \
--task translation \
--path $BASE_MODEL \
--model-overrides "{'eval_bleu': False, 'required_seq_len_multiple':1, 'load_alignments': False}" \
--dataset-impl mmap \
--valid-subset train \
--skip-invalid-size-inputs-valid-test \
--max-tokens 4096 \
--bpe fastbpe \
--user-dir $PROJECT_PATH/knnbox/models \
--arch vanilla_knn_mt@transformer_wmt19_de_en\
--knn-mode build_datastore \
--knn-datastore-path $DATASTORE_SAVE_PATH