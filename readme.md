# SW중심대학 공동 AI 경진대회 2023
# 4등상(전체 9등)

### team 0601p

model descriptions and train methods are in 0601p_발표자료

https://dacon.io/competitions/official/236092/overview/description

### preprocess, install deps
download open.zip from
https://dacon.io/competitions/official/236092/data
```
unzip open.zip -d ./input/
pip install -r requirements.txt
python dataset.py --mode train
# this is to install ops_dcnv3
./start.sh
```

### train code
```
train.py --augment True --loss torch.nn.BCEWithLogitsLoss --dataset_path ./input/ --lr 0.00004 --max_lr 0.0004 --model_name UperNet_InternImage_B --ckpt_dir ./checkpoint/ --batch_size 50 --epochs 700 --last_save True
```

### visualization code
use visualize.ipynb to visualize the model output. 