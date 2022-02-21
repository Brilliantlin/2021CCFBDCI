#!/usr/bin/env bash
cd code1
pwd
python dataprepare.py
echo "开始训练！"
python run_att.py
python run_multitask.py
echo "开始预测！"
python infer_att_cv.py --model_name attention_fgm  --input_file  ./data_new/cuted_testB.csv --result_file ../prediction_result/attention_fgm_tta --threshold 0.3
python infer_multitask.py --model_name  mutitask --input_file  ./data_new/cuted_testB.csv --result_file ../prediction_result/mutitask --threshold 0.5
python post2.py
