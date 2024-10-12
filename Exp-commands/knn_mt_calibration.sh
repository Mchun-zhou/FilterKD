#2.knnmt推理：       it(8,0.7,10,20)    law、medical(4,0.8,10,20)  koran(16,0.8,100,20)
# +knn分布校准：利用源语言的相似性对距离进行加权： it:46.59     law:62.02       medical:55.11       koran:20.57
PROJECT_PATH=
BASE_MODEL=
DATASET=it
DATA_PATH=
DATASTORE_LOAD_PATH=
CUDA_VISIBLE_DEVICES=0 python $PROJECT_PATH/knnbox-scripts/common/generate.py $DATA_PATH \
--task translation \
--path $BASE_MODEL \
--dataset-impl mmap \
--beam 4 --lenpen 0.6 --max-len-a 1.2 --max-len-b 10 --source-lang de --target-lang en \
--gen-subset valid \
--max-tokens 4096 \
--scoring sacrebleu \
--tokenizer moses \
--remove-bpe \
--user-dir $PROJECT_PATH/knnbox/models \
--arch vanilla_knn_mt@transformer_wmt19_de_en \
--knn-mode inference \
--knn-datastore-path $DATASTORE_LOAD_PATH \
--knn-k 8 \
--knn-lambda 0.7 \
--knn-temperature 10.0 \
--knn-u 20.0