from runconfig import *
from utils.config import loadConfigFromYml
from utils.dict2Obj import Dict2Obj

configs = loadConfigFromYml('config.yaml')
data_config = Dict2Obj(configs['data_config'])
model_config = Dict2Obj(configs['model_config'])
configs = Dict2Obj(configs)
TOKEN_MASK_SHAPE = (1,768)

from functools import partial
import paddle
import paddle.nn.functional as F
import paddlenlp as ppnlp
from paddlenlp.data import Tuple, Pad
from data import create_dataloader, QMSet,convert_example_with_attention_token
from model import QuestionMatchingAttention
from paddlenlp.datasets import MapDataset
from utils.myfile import loadPkl
from rule import *

def getRuleTest(test_path,test_sub):
    '''
    获取规则结果
    :param test_path:
    :return:
    '''
    print('🐕🐕🐕🐕🐕🐕🐕🐕🐕🐕🐕🐕🐕🐕🐕🐕🐕🐕🐕 read test data: %s ' % (test_path))
    test_data = pd.read_csv(test_path, sep='\t', header=None)
    test_data.columns = ['text_a', 'text_b']
    test_data['label'] = test_sub
    test_data = process(test_data)
    test_data = rule(test_data)
    return test_data

def batchify_fn(samples):
    inputs = Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # text_pair_input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # text_pair_segment
        Pad(axis=0, pad_val=np.zeros(TOKEN_MASK_SHAPE)),
    )(samples)
    keys = ['input_ids', 'token_type_ids', 'select_tokens',]
    batch_inputs = {k: v for k, v in zip(keys, inputs)}
    return batch_inputs

def predict(model,data_loader):
    '''
    预测逻辑
    :param model:
    :param data_loader:
    :return:
    '''
    model.eval()
    batch_logits = []
    total_num = 0
    with paddle.no_grad():
        for batch in tqdm(data_loader,disable=True):
            logits = model(**batch)
            logits = F.sigmoid(logits)
            batch_logits.append(logits.numpy())

    batch_logits = np.concatenate(batch_logits, axis=0)
    preds = batch_logits[:, 1]
    # preds = batch_logits.argmax(axis=1)
    return preds

def inferBertClassify(test_loader,model_structure,model_save_dir):
    '''

    Args:
        batch_size (int):
        test: test_set.Features
        model_structure: 模型结构，还需要load才能预测
        model_save_dir: 模型路径目录

    Returns:

    '''
    model_num = 0
    res = []
    t = os.listdir(model_save_dir)
    t= sorted(t)
    for model_state_file in tqdm(t,desc='预测中'):
        model = model_structure
        model.load(model_save_dir + model_state_file)
        tmp = predict(model, test_loader)
        print(tmp)
        res.append(tmp)
    return res



# --model_name
# mark_attention
# --input_file
# ../../data/test_A.tsv
# --result_file
# ../../prediction_result/mark_attention
#

if __name__ == "__main__":

    # TEST_PATH = '../data/test_B_1118.tsv'
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="The full path model file")
    parser.add_argument("--input_file", type=str, required=True, help="The full path of input file")
    parser.add_argument("--result_file", type=str, required=True, help="The result file name")
    parser.add_argument('--threshold', default=0.3, type=float,
                        help="Select which device to train model, defaults to gpu.")
    parser.add_argument('--batch_size', default=512, type=int, help="infer batch size")
    args = parser.parse_args()

    tokenizer = ppnlp.transformers.ErnieGramTokenizer.from_pretrained(
        model_config.init_ckpt)
    pretrained_model = ppnlp.transformers.ErnieGramModel.from_pretrained(
        model_config.init_ckpt)

    paddle.set_device(DEVICE)

    trans_func = partial(
        convert_example_with_attention_token,
        tokenizer=tokenizer,
        max_seq_length=data_config.max_seq_length,
        is_test=True,
        ratio=data_config.ratio)

    test_df = pd.read_csv(args.input_file,error_bad_lines=False)
    # if test_df.shape[1] == 3:
    #     test_df.columns = ['text_a','text_b','label']
    # elif test_df.shape[1] == 2:
    #     test_df.columns = ['text_a', 'text_b']
    # else:
    #     raise Exception

    test_ds = MapDataset(QMSet(test_df,choice=1))
    test_data_loader = create_dataloader(
        test_ds,
        mode='predict',
        batch_size=args.batch_size,
        batchify_fn=batchify_fn,
        trans_fn=trans_func)

    infer_model_config_path = './user_data/configs/' + args.model_name + '_' + 'config.pkl'
    model_config = loadPkl(infer_model_config_path)
    model = QuestionMatchingAttention(pretrained_model, model_config)
    model_save_dir = model_config.save_dir
    res = inferBertClassify(test_data_loader, model, model_save_dir)

    tta = True
    if tta:
        tmp = test_df['text_a'].copy()
        test_df['text_a'] =  test_df['text_b'].copy()
        test_df['text_b'] =  tmp
        test_ds = MapDataset(QMSet(test_df, choice=1))
        test_data_loader = create_dataloader(
            test_ds,
            mode='predict',
            batch_size=args.batch_size,
            batchify_fn=batchify_fn,
            trans_fn=trans_func)
        res2 = inferBertClassify(test_data_loader, model, model_save_dir)
        res_tta = res + res2


    print('####################################1.概率融合####################################')
    prob_ensemble = np.mean(res_tta,axis=0)
    # np.save('best_prob_899.npy',prob_ensemble)
    np.save('../prediction_result/att_prob.npy',prob_ensemble)
    print('阈值划分',args.threshold)
    y_preds = np.where(prob_ensemble > args.threshold, 1, 0)
    print('总计正样本数：',np.sum(y_preds))
    # test_data = getRuleTest(TEST_PATH,test_sub = y_preds)
    # print('纠正后正样本数：',test_data['label'].sum())\
    test_df['label'] = y_preds
    test_df['label'].to_csv(args.result_file + '_%s.csv' % (args.threshold), header=None, index=None)

    print('####################################2.投票融合####################################')
    threshold = [args.threshold] * 3
    res_all = [ np.where(i > args.threshold, 1, 0) for i in res]
    print('各模型预测个数分别为：')
    for i in res_all:
        print(i.sum(),sep=' ')
    res_num = len(res)
    v = res_num//2 if res_num/2 == int(res_num)//2 else (res_num//2) + 1

    print('票数：%s' % (v))
    vote_res  = [ 1 if vote > v  else 0 for vote in np.sum(res_all, axis=0)]
    print('总计正样本数：', np.sum(vote_res))
    # test_data = getRuleTest(TEST_PATH, test_sub=vote_res)
    # print('纠正后正样本数：', test_data['label'].sum())
    test_df['label'] = vote_res
    test_df['label'].to_csv(args.result_file + '_%s_vote_%s_.csv' % ('_'.join([str(x) for x in threshold]),v), header=None, index=None)

