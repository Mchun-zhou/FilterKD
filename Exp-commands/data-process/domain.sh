#!/bin/bash
#目录：/home/zmc/data_process/ ;conda环境为：dataprocess ,python=3.7,pytorch=1.12.0
#执行 bash prepare-domain-adapt.sh koran


DAMAIN= #数据集标识
DATADIR=
BPEDATA= #bpe处理后的文件目录
HOME= #项目地址
if [ -z $HOME ]
then
  echo "HOME var is empty, please set it"
  exit 1
fi
SCRIPTS=$HOME/mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
FASTBPE=$HOME/fastBPE
BPECODES= # 预训练模型学到的codes 可从官方找到对应的.codes文件
VOCAB= #词表也是 因为共享此表 所以选择dict.de.txt / dict.en.txt都可以

if [ ! -d "$SCRIPTS" ]; then
  echo "Please set SCRIPTS variable correctly to point to Moses scripts."
  exit
fi

mkdir ${BPEDATA}

filede=${DATADIR}/train.de
fileen=${DATADIR}/train.en

cat $filede | \
  perl $NORM_PUNC $l | \
  perl $REM_NON_PRINT_CHAR | \
  perl $TOKENIZER -threads 8 -a -l de  >> ${BPEDATA}/train.tok.de

cat $fileen | \
  perl $NORM_PUNC $l | \
  perl $REM_NON_PRINT_CHAR | \
  perl $TOKENIZER -threads 8 -a -l en  >> ${BPEDATA}/train.tok.en

$FASTBPE/fast applybpe ${BPEDATA}/train.bpe.de ${BPEDATA}/train.tok.de $BPECODES $VOCAB
$FASTBPE/fast applybpe ${BPEDATA}/train.bpe.en ${BPEDATA}/train.tok.en $BPECODES $VOCAB

perl $CLEAN -ratio 1.5 ${BPEDATA}/train.bpe de en ${BPEDATA}/train 1 250

for split in valid test
do
  filede=${DATADIR}/${split}.de
  fileen=${DATADIR}/${split}.en

  cat $filede | \
    perl $TOKENIZER -threads 8 -a -l de  >> ${BPEDATA}/${split}.tok.de

  cat $fileen | \
    perl $TOKENIZER -threads 8 -a -l en  >> ${BPEDATA}/${split}.tok.en

  $FASTBPE/fast applybpe ${BPEDATA}/${split}.de ${BPEDATA}/${split}.tok.de $BPECODES $VOCAB
  $FASTBPE/fast applybpe ${BPEDATA}/${split}.en ${BPEDATA}/${split}.tok.en $BPECODES $VOCAB
done