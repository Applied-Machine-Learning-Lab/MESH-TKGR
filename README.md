Code implementation for MESH

Run the experiments by: 
```
cd ./src
python main_llm_mmoe_norm.py -d ICEWS14s --train-history-len 3 --test-history-len 3 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --gpu=0 --n-hidden 100 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5  --entity-prediction --relation-prediction --add-static-graph --angle 10 --discount 1 --task-weight 0.7 --weight_loss 1
```
