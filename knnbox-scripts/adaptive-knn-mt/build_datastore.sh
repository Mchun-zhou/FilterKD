PROJECT_PATH=/home/mc/adapter-kd/
BASE_MODEL="/home/mc/prepare/pretrain-models/wmt19.de-en/wmt19.de-en.ffn8192.pt"
DATA_PATH=/home/mc/DATA/it
DATASTORE_SAVE_PATH=$PROJECT_PATH/datastore/vanilla/it

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
--arch adaptive_knn_mt@transformer_wmt19_de_en \
--knn-mode build_datastore \
--knn-datastore-path  $DATASTORE_SAVE_PATH \
