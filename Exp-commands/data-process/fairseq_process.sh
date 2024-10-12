# path：bpe后的数据位置以及保存后二值化的数据目录、以及字典位置
python  ./fairseq_cli/preprocess.py \
--source-lang de --target-lang en \
--trainpref  /path/train \  #/path/bpe后的数据位置
--validpref /path/valid \
--testpref /path/test \
--destdir /path/data-bin \
--srcdict path/dict.de.txt \
 --joined-dictionary --workers 8