
DATA_PATH=  #数据集加载位置
MODEL_PATH= #模型保存位置
CUDA_VISIBLE_DEVICES=0 python train.py $DATA_PATH \
    --task translation \
    --arch transformer_iwslt_de_en \
    --share-all-embeddings \
    --optimizer adam --lr 0.0005 -s en -t de \
    --label-smoothing 0.1 --dropout 0.3 --max-tokens 8192 \
    --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy \
    --no-progress-bar \
    --fp16 \
    --max-update 300000 --warmup-updates 4000 --warmup-init-lr 1e-07 --adam-betas '(0.9,0.98)' \
    --keep-last-epochs 2 \
    --log-interval 100 --save-interval-updates 1000 --keep-interval-updates 10 --save-interval 10000\
    --save-dir $MODEL_PATH 2>&1 | tee $MODEL_PATH/training.log