

### 🛠️ 环境配置
代码基于 [fairseq](https://github.com/facebookresearch/fairseq/tree/main) 和d knnbox工具包.

环境配置 :
```shell
git clone 该项目
conda create -n your_project python=3.7
conda activate 
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
conda install faiss-gpu -c pytorch
#cd 项目路径下,配置fairseq
pip install --editable ./

#其他依赖包：
pip install -r requirements.txt
```

### 数据获取以及预处理
对于多领域数据可从[这里](https://github.com/roeeaharoni/unsupervised-domain-clusters)获取；
低资源越南语-英语和土耳其语-英语可从[这里](https://nlp.stanford.edu/projects/nmt/)获取；
对于预训练的wmt19 de-en模型可在[fairseq/example/translation]()目录下找到下载链接；

```bash
# 1.数据bpe处理：
bash Exp-commands/data-process/domain.sh
# 2.fairseq二进制处理：
bash Exp-commands/data-process/fairseq_process.sh
```

### 模型训练
多领域数据集可直接使用预训练模型；
低资源语言对首先训练标准的transformer:
```shell
bash Exp-commands/train-base-transformer.sh
```

```shell
#创建数据存储
bash Exp-commands/bash_build_ds.sh
#网格搜索寻找最佳超参数：
bash Exp-commands/find_params.sh
#标准的knn-mt推理：
bash Exp-commands/knn_mt.sh
#校准后的knn-mt测试：
bash Exp-commands/knn_mt_calibration.sh
#训练模型：
bash train.sh
```

### 📏 模型评估
```shell
bash inference.sh
```

