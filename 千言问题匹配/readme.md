
# 赛题简介


# 方案简介

本团队将该任务转为自然语言处理中常见的二分类任务，主要选择使用了Ernie-Gram、NeZha等最新的预训练模型。针对该题句对样本之间差距小这一难点，以增大句对样本间差异为切入点，用词性标注模型提取词性特征，引入TOKEN-MASK机制指导模型重点学习两句子的差异部分。由于本题数据集来源不同，属于一个跨领域问题，本团队将数据集分类任务作为辅助任务，使用多任务学习来适应不同数据集之间的标注偏差。此外本团队还使用了随机对偶增强、FGM对抗训练、TTA(Test-Time Augmentation)、SWA等技术增强模型的鲁棒性。最终通过Ranking的集成方式进行集成，结合拼音信息、中文词性标注信息、中文近反义词典对模型难以学习的语法模式进行规则矫正。在千言问题鲁棒性匹配评测任务上，复赛成绩92.72，排名第一。



## 数据预处理： 
    
- 使用百度词性标注模型 分词、词性标注
- 地理位置识别
- 中文、阿拉伯数字统一表示
- 拼音识别
- 细粒度变化识别（插入、替换、位置调换）


## 模型一 （nezha-large+TTA+阈值搜索+fgm+5折 887）

传统的BERT + Linear 分类结构，在bert输入时对数据做对偶，前向时不同数据对做两次model，该模型对对话理解项指标有显著优势

## 模型二（ernie-gram-zh 单模  + fgm + 阈值搜索 + TTA 891）：
    
针对本赛题数据的特点。以编辑距离为划分指标，对于编辑距离大于70的样本，只选择两句中不同的字符token做mean-pooling，学习细粒度知识。对于编辑距离小于70的样本，对所有字符token做mean-pooling、
            
## 模型三  （多任务 单模 + fgm + 阈值搜索 + TTA 887）：

针对不同领域的标注原则有些可能不一样这一问题，在模型一的基础上增加领域分类任务，以帮助模型区分数据领域及对应的标签关系

模型二和模型三在词汇理解和语法结构两项指标上有显著优势。


## 模型后处理

- 同义词纠正：检测出两句子发生替换操作，且不是询问组词、造句、翻译、拼读写，且替换词语为同义词，置为正样本。
- 反义词纠正：检测出两句子发生替换操作，且不是询问组词、造句、翻译、拼读写，且替换词语为反义词，置为负样本。
- 并列关系纠正：检测句子发生交换词语操作，且为并列关系（中间词为表并列关系的词），置为正样本。
- 地名实体交换1: 检测句子发生地名实体的交换，且为交换不改变意思的对称结构，置为正样本。
- 地名实体交换2： 检测句子发生地名实体的交换，且为交换改变意思的结构，置为负样本。
- 主谓替换：检测句子发生事件主体的替换，且交换后意思颠倒（中间词大多为动词），置为负样本。
- 插入词语处理：检测句子发生插入操作，将插入颜色、程度、状态等词语的以及置性度低的样本置0。
- 拼写错误处理：检测句子对只有1个字不一样，且拼音相同置为负样本。（排除询问拼、读写、意思的情况）
- 询问组词类处理：提取需要组词的字主题，比较主题是否相等，相等的置为正样本不相等的置为负样本。
- 其他样本：三套模型方案加权rank融合。


## 其他尝试

- 数据增强会引入噪声，不适合本题
- 对于模型一、二结构5折收益不高，因此采用单折
- 加入词性、句法特征，训练bi-gru后拼接到ernie-gram词向量，主要是想添加特征增加两句的差异度，单模大概在0.883左右。比较依赖词性标注的结果，有误差传播，没有加入最终方案融合。
- Prompt 模板，在对话理解单项上具有优势，其他项一般，未加入最终融合方案

# 代码说明
## 代码结构
```
    angular2html
    |-- B
        |-- Dockerfile
        |-- __init__.py
        |-- docker_build.sh
        |-- infer_lin.sh #
        |-- readme.md
        |-- requirements.txt
        |-- run.sh
        |-- run_lin.sh
        |-- run_xia.sh
        |-- .ipynb_checkpoints
        |-- code1                #模型二和模型三 paddle实现
        |   |-- __init__.py 
        |   |-- config.yaml         #一些配置文件
        |   |-- config_add_lac.yaml
        |   |-- config_mutitask.yaml
        |   |-- data.py                  #数据读取和数据的转换
        |   |-- dataprepare.py           #数据准备、主要是提取词性和分词
        |   |-- infer_att_cv.py          #模型二 推理
        |   |-- infer_att_lac.py 
        |   |-- infer_multitask.py       #模型二推理
        |   |-- model.py                 #模型文件 所有模型都在这
        |   |-- post2.py                 # 后处理代码
        |   |-- rule.py            
        |   |-- run_att.py               #模型一训练
        |   |-- run_att_lac.py   
        |   |-- run_multitask.py         #模型三训练
        |   |-- runconfig.py
        |   |-- train.py           #训练代码
        |   |-- data_new
        |   |   |-- cuted_testB.csv #dataprepare 生成的测试集数据
        |   |   |-- gaiic_track3_round1_train_20210220.tsv
        |   |   |-- new_test.csv   #dataprepare 生成的测试集数据
        |   |   |-- new_testB.csv  #dataprepare 生成的测试集数据
        |   |   |-- new_train.csv  #dataprepare 生成的测试集数据
        |   |   |-- test_.csv  
        |   |   |-- 反义词库.txt    #来源于网络 + 训练集统计修正
        |   |   |-- 否定词库.txt    #来源于网络 + 训练集统计修正
        |   |   |-- 新同义词典.txt  #来源于网络 + 训练集统计修正
        |   |-- user_data
        |   |   |-- configs
        |   |   |   |-- attention_fgm_config.pkl
        |   |   |   |-- mutitask_config.pkl
        |   |   |-- models          #存放两Ernie方案的模型
        |   |       |-- attention_fgm 
        |   |       |   |-- best_val_stepsingle.pdparams
        |   |       |   |-- best_val_stepsingle_23700.pdparams
        |   |       |   |-- best_val_stepsingle_24000.pdparams
        |   |       |-- mutitask    
        |   |           |-- best_val_stepsingle.pdparams
        |   |           |-- best_val_stepsingle_18300.pdparams
        |   |           |-- best_val_stepsingle_20700.pdparams
        |   |           |-- best_val_stepsingle_22500.pdparams
        |   |           |-- best_val_stepsingle_24000.pdparams
        |   |-- utils               #一些工具 有些用到有些没用到
        |       |-- BaseModel.py
        |       |-- __init__.py
        |       |-- attack.py
        |       |-- config.py
        |       |-- dict2Obj.py
        |       |-- log_setting.py
        |       |-- myfile.py
        |       |-- seed.py
        |       |-- threshold.py
        |-- code2
        |   |-- __init__.py
        |   |-- locate_main.py  #关键词地域识别
        |   |-- new_jugment2.py  #数据swap部分判别
        |   |-- nezha.py  
        |   |-- post1.py   #后处理1
        |   |-- predict_nezha.py   #nezha权重预测
        |   |-- torch_nezha_large.py  #nezha模型训练
        |   |-- dict #全国区县字典，来源网络以及搜集信息
        |       |-- __init__.py
        |       |-- china_city_dict.txt
        |       |-- china_county_dict.txt
        |       |-- china_ns.txt
        |       |-- china_province_dict.txt
        |       |-- confuse_county_dict.txt
        |       |-- confused_city_dict.txt
        |       |-- confused_county_dict.txt
        |       |-- new_china_ns_dict.txt
        |       |-- outside_dict.py
        |-- ernie-gram-zh
        |   |-- ernie_gram_zh.pdparams
        |   |-- model_config.json
        |   |-- model_state.pdparams
        |   |-- vocab.txt
        |-- logs   # 存放训练日志
        |-- prediction_result #存放结果文件夹
        |-- raw_data
        |   |-- test_A.tsv
        |   |-- test_B_1118.tsv
        |   |-- BQ
        |   |   |-- dev
        |   |   |-- test
        |   |   |-- train
        |   |-- LCQMC
        |   |   |-- dev
        |   |   |-- test
        |   |   |-- train
        |   |-- OPPO
        |       |-- dev
        |       |-- train
        |-- user_data
            |-- data_test  #生成的中间文件
            |   |-- test21.csv
            |   |-- test_B_region20.csv
            |   |-- .ipynb_checkpoints
            |-- nezha-large  #nezha-large原始权重
            |   |-- bert_config.json
            |   |-- config.json
            |   |-- pytorch_model.bin
            |   |-- vocab.txt
           
```



