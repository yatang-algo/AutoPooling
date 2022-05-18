# AutoPooling: Automated Pooling Search for Multi-valued Features in Recommendations

## environment
- python 3.6.13
- torch 1.6.0

## train model


```
cd scripts
python base_dnn.py  #dnn base model
python auto_dnn_hard_search.py   #autopooling based on dnn, hard version 
python auto_dnn_hybrid.py  #autopooling based on dnn, hybrid version 

```
