# Classificação de Lixo Utilizando Redes Neurais

## Requerimentos

- Pytorch
- Matplotlib
- Numpy
- Scikit-learn

## Execução

```
python3 main.py  [-h] [-d {GarbageClass}] [--log-dir LOG_DIR] [--model {resnet-18,resnet-101,inception-v3}] [--lr LR]
                 [--pretrained] [--max-epoch MAX_EPOCH] [--seed SEED] [--batch-size BATCH_SIZE]
                 [--resume PATH] [--mode {train,test,plot,testOnTrain}]

  -h, --help            show this help message and exit
  -d {GarbageClass}, --dataset {GarbageClass}
                        options: dict_keys(['GarbageClass'])
  --log-dir LOG_DIR     Directory to store logs and trained models
  --model {resnet-18,resnet-101,inception-v3}
                        Options: dict_keys(['resnet-18', 'resnet-101', 'inception-v3'])
  --lr LR               Learning Rate
  --pretrained
  --max-epoch MAX_EPOCH
  --seed SEED           manual seed for random
  --batch-size BATCH_SIZE
  --resume PATH         root directory where previous train is saved
  --mode {train,test,plot,testOnTrain}
                        Options: [train, test, plot, testOnTrain]
```