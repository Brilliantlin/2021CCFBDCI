
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddlenlp as ppnlp

from utils.BaseModel import BasicModule

class LacLayer(BasicModule):
    def __init__(self,
                 weight_attr,
                 vocab_size=49,
                 emb_dim=4,
                 padding_idx=0,
                 gru_hidden_size=198,
                 direction='bidirect',
                 gru_layers=1,
                 dropout_rate=0.1,
                 ):
        super(LacLayer, self).__init__()
        self.embedder = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=emb_dim,
            padding_idx=padding_idx,
            weight_attr=weight_attr,
        )

        self.gru_layer = nn.GRU(input_size=emb_dim,
                                hidden_size=gru_hidden_size,
                                num_layers=gru_layers,
                                direction=direction,
                                dropout=dropout_rate,
                                weight_hh_attr= weight_attr,
                                weight_ih_attr= weight_attr,
                                )

    def forward(self, lac_ids, sequence_length, **kwargs):
        embedded_text = self.embedder(lac_ids)
        encoded_text, last_hidden = self.gru_layer(
            embedded_text, sequence_length=sequence_length)
        return encoded_text


class DepLayer(BasicModule):
    def __init__(self,
                 weight_attr,
                 vocab_size=29,
                 emb_dim=4,
                 padding_idx=0,
                 gru_hidden_size=198,
                 direction='bidirect',
                 gru_layers=1,
                 dropout_rate=0.1,
                 ):
        super(DepLayer, self).__init__()
        self.embedder = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=emb_dim,
            padding_idx=padding_idx,
            weight_attr=weight_attr,
        )
        self.gru_layer = nn.GRU(input_size=emb_dim,
                                hidden_size=gru_hidden_size,
                                num_layers=gru_layers,
                                direction=direction,
                                dropout=dropout_rate,
                                weight_hh_attr=weight_attr,
                                weight_ih_attr=weight_attr,
                                )

    def forward(self, lac_ids, sequence_length, **kwargs):
        embedded_text = self.embedder(lac_ids)
        encoded_text, last_hidden = self.gru_layer(
            embedded_text, sequence_length=sequence_length)
        return encoded_text


class QuestionMatchingAttentionAddFeature(BasicModule):
    def __init__(self, pretrained_model, config):
        super(QuestionMatchingAttentionAddFeature, self).__init__()
        self.ptm = pretrained_model
        self.config = config
        self.dropout = nn.Dropout(config.dropout if config.dropout is not None else 0.15)

        weight_attr = paddle.ParamAttr(
                                       learning_rate=20,  # 全局学习率的倍数
                                       trainable=True)

        self.lac_layer = LacLayer(
            weight_attr=weight_attr,
            vocab_size=config.lac_vocab_size,
            emb_dim=config.gru_emb_dim,
            padding_idx=0,
            gru_hidden_size=config.gru_hidden_size,
            direction='bidirectional' if config.direction == 2 else 'forward',
            gru_layers=config.gru_layers,
            dropout_rate=config.gru_dropout_rate)

        self.dep_layer = DepLayer(vocab_size=config.dep_vocab_size,
                                  weight_attr=weight_attr,
                                  emb_dim=config.gru_emb_dim,
                                  padding_idx=0,
                                  gru_hidden_size=config.gru_hidden_size,
                                  direction='bidirectional' if config.direction == 2 else 'forward',
                                  gru_layers=config.gru_layers,
                                  dropout_rate=config.gru_dropout_rate)

        self.classifier = nn.Linear(self.ptm.config["hidden_size"] + config.gru_hidden_size * config.direction * 2, 2)
        self.rdrop_coef = config.rdrop_coef
        self.rdrop_loss = ppnlp.losses.RDropLoss()

    def forward(self,
                input_ids,
                token_type_ids,
                select_tokens,
                lac_ids,
                dep_ids,
                sequence_length,
                position_ids=None,
                attention_mask=None,
                do_evaluate=False,
                **params):
        batch_shape = len(select_tokens)  # [bs,index]
        hidden_state, cls_embedding1 = self.ptm(input_ids, token_type_ids, position_ids,  # [bs,max_len,768]
                                                attention_mask)
        lac_hidden = self.lac_layer(lac_ids, sequence_length=sequence_length)
        dep_hidden = self.dep_layer(dep_ids, sequence_length=sequence_length)
        select_length = select_tokens.sum(axis=1)[:, 0].detach()  # 每个样本的mask后的长度
        hidden_state = paddle.concat([hidden_state, lac_hidden, dep_hidden], axis=-1)

        x = hidden_state * select_tokens.detach()  #
        tmp_sum = paddle.sum(x, axis=1)
        x = tmp_sum / (select_length.reshape((batch_shape, 1)))
        logits1 = self.classifier(x)

        # For more information about R-drop please refer to this paper: https://arxiv.org/abs/2106.14448
        # Original implementation please refer to this code: https://github.com/dropreg/R-Drop
        if self.rdrop_coef > 0 and not do_evaluate:
            _, cls_embedding2 = self.ptm(input_ids, token_type_ids, position_ids,
                                         attention_mask)
            cls_embedding2 = self.dropout(cls_embedding2)
            logits2 = self.classifier(cls_embedding2)
            kl_loss = self.rdrop_loss(logits1, logits2)
        else:
            kl_loss = 0.0

        return logits1


class QuestionMatching_Siamese(BasicModule):
    '''
    孪生模型
    '''

    def __init__(self, pretrained_model, config):
        super(QuestionMatching_Siamese, self).__init__()

        self.ptm1 = pretrained_model
        # self.ptm2 = deepcopy(pretrained_model)
        self.ptm2 = pretrained_model

        self.config = config
        self.dropout = nn.Dropout(config.dropout if config.dropout is not None else 0.15)
        # num_labels = 2 (similar or dissimilar)
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.ptm1.config["hidden_size"], 2),
        )
        self.rdrop_coef = config.rdrop_coef
        self.rdrop_loss = ppnlp.losses.RDropLoss()

    def forward(self,
                input_ids1,
                token_type_ids1,
                select_index1,
                input_ids2,
                token_type_ids2,
                select_index2,
                attention_mask=None,
                do_evaluate=False,
                **params):
        hidden1, u = self.ptm1(input_ids1, token_type_ids1, attention_mask=None)  # [bn,max_len,768]
        hidden2, v = self.ptm2(input_ids2, token_type_ids2, attention_mask=None)

        u_v = paddle.abs(u - v)
        # x = paddle.concat([u,v,u_v],axis=-1) #[bn,768*3]
        logits1 = self.classifier(u_v)
        return logits1


class QuestionMatching(BasicModule):

    def __init__(self, pretrained_model, config):
        super(QuestionMatching, self).__init__()
        self.ptm = pretrained_model
        self.config = config
        self.dropout = nn.Dropout(config.dropout if config.dropout is not None else 0.15)

        # num_labels = 2 (similar or dissimilar)
        self.classifier = nn.Linear(self.ptm.config["hidden_size"], 2)
        self.rdrop_coef = config.rdrop_coef
        self.rdrop_loss = ppnlp.losses.RDropLoss()

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None,
                do_evaluate=False,
                **params):

        _, cls_embedding1 = self.ptm(input_ids, token_type_ids, position_ids,
                                     attention_mask)
        cls_embedding1 = self.dropout(cls_embedding1)
        logits1 = self.classifier(cls_embedding1)

        # For more information about R-drop please refer to this paper: https://arxiv.org/abs/2106.14448
        # Original implementation please refer to this code: https://github.com/dropreg/R-Drop
        if self.rdrop_coef > 0 and not do_evaluate:
            _, cls_embedding2 = self.ptm(input_ids, token_type_ids, position_ids,
                                         attention_mask)
            cls_embedding2 = self.dropout(cls_embedding2)
            logits2 = self.classifier(cls_embedding2)
            kl_loss = self.rdrop_loss(logits1, logits2)
        else:
            kl_loss = 0.0

        return logits1 + kl_loss


class QuestionMatchingAttention(BasicModule):
    def __init__(self, pretrained_model, config):
        super(QuestionMatchingAttention, self).__init__()
        self.ptm = pretrained_model
        self.config = config
        self.dropout = nn.Dropout(config.dropout if config.dropout is not None else 0.15)

        # num_labels = 2 (similar or dissimilar)
        self.classifier = nn.Linear(self.ptm.config["hidden_size"], 2)
        self.rdrop_coef = config.rdrop_coef
        self.rdrop_loss = ppnlp.losses.RDropLoss()

    def forward(self,
                input_ids,
                token_type_ids,
                select_tokens,
                position_ids=None,
                attention_mask=None,
                do_evaluate=False,
                **params):
        batch_shape = len(select_tokens)  # [bs,index]
        hidden_state, cls_embedding1 = self.ptm(input_ids, token_type_ids, position_ids,  # [bs,max_len,768]
                                                attention_mask)

        select_length = select_tokens.sum(axis=1)[:, 0].detach()  # 每个样本的mask后的长度
        x = hidden_state * select_tokens.detach()  #

        tmp_sum = paddle.sum(x, axis=1)
        x = tmp_sum / (select_length.reshape((batch_shape, 1)))
        logits1 = self.classifier(x)

        # For more information about R-drop please refer to this paper: https://arxiv.org/abs/2106.14448
        # Original implementation please refer to this code: https://github.com/dropreg/R-Drop
        if self.rdrop_coef > 0 and not do_evaluate:
            _, cls_embedding2 = self.ptm(input_ids, token_type_ids, position_ids,
                                         attention_mask)
            cls_embedding2 = self.dropout(cls_embedding2)
            logits2 = self.classifier(cls_embedding2)
            kl_loss = self.rdrop_loss(logits1, logits2)
        else:
            kl_loss = 0.0

        return logits1 + kl_loss * self.rdrop_coef

class QuestionMatchingAttentionDomain(BasicModule):
    def __init__(self, pretrained_model, config):
        super(QuestionMatchingAttentionDomain, self).__init__()
        self.ptm = pretrained_model
        self.config = config
        self.dropout = nn.Dropout(config.dropout if config.dropout is not None else 0.15)

        # num_labels = 2 (similar or dissimilar)
        self.classifier = nn.Linear(self.ptm.config["hidden_size"], 2)
        self.classifier_domain = nn.Sequential(nn.Dropout(0.2),
                                               nn.Linear(self.ptm.config["hidden_size"], 3),
                                               )
        self.rdrop_coef = config.rdrop_coef
        self.rdrop_loss = ppnlp.losses.RDropLoss()

    def forward(self,
                input_ids,
                token_type_ids,
                select_tokens,
                position_ids=None,
                attention_mask=None,
                do_evaluate=False,
                **params):
        batch_shape = len(select_tokens)  # [bs,index]
        hidden_state, cls_embedding1 = self.ptm(input_ids, token_type_ids, position_ids,  # [bs,max_len,768]
                                                attention_mask)

        select_length = select_tokens.sum(axis=1)[:, 0].detach()  # 每个样本的mask后的长度
        x = hidden_state * select_tokens.detach()  #

        tmp_sum = paddle.sum(x, axis=1)
        x = tmp_sum / (select_length.reshape((batch_shape, 1)))
        logits1 = self.classifier(x)


        domain_out = self.classifier_domain(cls_embedding1)


        # For more information about R-drop please refer to this paper: https://arxiv.org/abs/2106.14448
        # Original implementation please refer to this code: https://github.com/dropreg/R-Drop
        if self.rdrop_coef > 0 and not do_evaluate:
            _, cls_embedding2 = self.ptm(input_ids, token_type_ids, position_ids,
                                         attention_mask)
            cls_embedding2 = self.dropout(cls_embedding2)
            logits2 = self.classifier(cls_embedding2)
            kl_loss = self.rdrop_loss(logits1, logits2)
        else:
            kl_loss = 0.0

        return logits1 + kl_loss * self.rdrop_coef,domain_out

class Ernie_cnn(BasicModule):
    def __init__(self, pretrained_model, config):
        super().__init__()
        self.ptm = pretrained_model
        self.config = config
        self.convs1 = nn.LayerList(
            [nn.Conv2D(1, config.kernel_num, (f, self.ptm.config["hidden_size"])) for f in config.kernel_sizes])
        self.bns = nn.LayerList([nn.BatchNorm1D(config.kernel_num, ) for i in range(len(config.kernel_sizes))])
        self.simple_fc = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.kernel_num * len(config.kernel_sizes), config.num_labels),
        )

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None,
                do_evaluate=False):
        bert_out = self.ptm(input_ids, token_type_ids, position_ids,
                            attention_mask)
        last_hidden_state, pooled_out = bert_out[0], bert_out[1]
        x = last_hidden_state
        x = x.unsqueeze(1)  # (N, Ci, token_num, embed_dim)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, token_num) * len(kernel_sizes)]
        x = [bn(i) for bn, i in zip(self.bns, x)]
        x = [F.max_pool1d(i, i.shape[2]).squeeze(2) for i in x]  # [(N, Co) * len(kernel_sizes)]
        x = paddle.concat(x, 1)  # (N, Co * len(kernel_sizes))
        logits1 = self.simple_fc(x)
        return logits1


if __name__ == "__main__":
    # from runconfig import model_config

    # pretrained_model = ppnlp.transformers.ErnieGramModel.from_pretrained(
        # 'ernie-gram-zh')
    # tokenizer = ppnlp.transformers.ErnieGramTokenizer.from_pretrained(
        # 'ernie-gram-zh')
    # model = Ernie_cnn(pretrained_model, model_config)
    # model = QuestionMatching(pretrained_model, model_config)
    # model.save(name='debug.pdparams')
    # prefix = './user_data/models/' + model.config.model_name + '/'  # 模型前缀
    # model.load(prefix + 'debug.pdparams')
