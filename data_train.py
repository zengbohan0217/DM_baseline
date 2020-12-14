import os
import tensorflow as tf
import keras
import bert4keras
import numpy as np
import pandas as pd
import codecs
import data_deal as ddl
import logging
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open, ViterbiDecoder, to_array
from bert4keras.layers import ConditionalRandomField
from keras.layers import Dense, Bidirectional, LSTM, Dropout
from keras.models import Model
from tqdm import tqdm

maxlen = 250
epochs = 10
# batch_size = 8
bert_layers = 12
learing_rate = 3e-5  # bert_layers越小，学习率应该要越大
crf_lr_multiplier = 1500  # 必要时扩大CRF层的学习率 #500,1500

# bert配置
config_path = './publish/bert_config.json'
checkpoint_path = './publish/bert_model.ckpt'
dict_path = './publish/vocab.txt'


def load_data(filename):
    D = []
    with open(filename, encoding='utf-8') as f:
        f = f.read()
        for l in f.split('\n\n'):
            if not l:
                continue
            d, last_flag = [], ''
            for c in l.split('\n'):
                try:
                    char, this_flag = c.split(' ')
                except:
                    print(c)
                    continue
                if this_flag == 'O' and last_flag == 'O':
                    d[-1][0] += char
                elif this_flag == 'O' and last_flag != 'O':
                    d.append([char, 'O'])
                elif this_flag[:1] == 'B':
                    d.append([char, this_flag[2:]])
                else:
                    d[-1][0] += char
                last_flag = this_flag
            D.append(d)
    return D


# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# 类别映射

labels = ['name',
          'movie',
          'organization',
          'position',
          'company',
          'game',
          'book',
          'address',
          'government',
          'scene',
          'email',
          'mobile',
          'QQ',
          'vx']

id2label = dict(enumerate(labels))
label2id = {j: i for i, j in id2label.items()}
num_labels = len(labels) * 2 + 1


class data_generator(DataGenerator):

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, item in self.sample(random):
            token_ids, labels = [tokenizer._token_start_id], [0]
            for w, l in item:
                w_token_ids = tokenizer.encode(w)[0][1:-1]
                if len(token_ids) + len(w_token_ids) < maxlen:
                    token_ids += w_token_ids
                    if l == 'O':
                        labels += [0] * len(w_token_ids)
                    else:
                        B = label2id[l] * 2 + 1
                        I = label2id[l] * 2 + 2
                        labels += ([B] + [I] * (len(w_token_ids) - 1))
                else:
                    break
            token_ids += [tokenizer._token_end_id]
            labels += [0]
            segment_ids = [0] * len(token_ids)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


def bertmodel():
    model = build_transformer_model(
        config_path,
        checkpoint_path,
    )
    output_layer = 'Transformer-%s-FeedForward-Norm' % (bert_layers - 1)
    output = model.get_layer(output_layer).output
    output = Dense(num_labels)(output)  # 27分类

    CRF = ConditionalRandomField(lr_multiplier=crf_lr_multiplier)
    output = CRF(output)

    model = Model(model.input, output)
    #     model.summary()

    model.compile(
        loss=CRF.sparse_loss,
        optimizer=Adam(learing_rate),
        metrics=[CRF.sparse_accuracy]
    )
    return model, CRF


class NamedEntityRecognizer(ViterbiDecoder):
    """命名实体识别器
    """

    def recognize(self, text):
        tokens = tokenizer.tokenize(text)
        mapping = tokenizer.rematch(text, tokens)
        token_ids = tokenizer.tokens_to_ids(tokens)
        segment_ids = [0] * len(token_ids)
        token_ids, segment_ids = to_array([token_ids], [segment_ids])
        nodes = model.predict([token_ids, segment_ids])[0]
        labels = self.decode(nodes)
        entities, starting = [], False
        for i, label in enumerate(labels):
            if label > 0:
                if label % 2 == 1:
                    starting = True
                    entities.append([[i], id2label[(label - 1) // 2]])
                elif starting:
                    entities[-1][0].append(i)
                else:
                    starting = False
            else:
                starting = False

        return [(text[mapping[w[0]][0]:mapping[w[-1]][-1] + 1], l)
                for w, l in entities]


def evaluate(data):
    """评测函数
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    for d in tqdm(data):
        text = ''.join([i[0] for i in d])
        R = set(NER.recognize(text))  # 预测
        T = set([tuple(i) for i in d if i[1] != 'O'])  # 真实
        X += len(R & T)
        Y += len(R)
        Z += len(T)
    precision, recall = X / Y, X / Z
    f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall


class Evaluator(keras.callbacks.Callback):
    def __init__(self, valid_data, mode=0):
        self.best_val_f1 = 0
        self.valid_data = valid_data
        self.mode = mode  # k折的时候记录第几折

    def on_epoch_end(self, epoch, logs=None):
        trans = K.eval(CRF.trans)
        NER.trans = trans
        #         print(NER.trans)
        f1, precision, recall = evaluate(self.valid_data)
        # 保存最优
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            model.save_weights('./best_bilstm_model_{}.weights'.format(self.mode))
        logging.info(
            'valid:  f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (f1, precision, recall, self.best_val_f1)
        )


batch_size = 12

train_data = load_data(f'./train (1)/deal_data_train/train_0.txt')
valid_data = load_data(f'./train (1)/deal_data_test/val_0.txt')
model, CRF = bertmodel()
NER = NamedEntityRecognizer(trans=K.eval(CRF.trans), starts=[0], ends=[0])

evaluator = Evaluator(valid_data)
train_generator = data_generator(train_data, batch_size)

model.fit_generator(
    train_generator.forfit(),
    steps_per_epoch=len(train_generator),
    epochs=epochs,
    callbacks=[evaluator]
)


def test_predict(data, NER_):
    test_ner = []
    for text in tqdm(data):
        cut_text_list, cut_index_list = ddl.cut_test_set([text], maxlen)    # cut_text_list中只有一句话
        posit = 0
        item_ner = []
        index = 1
        for str_ in cut_text_list:
            ner_res = NER_.recognize(str_)    # 输入一句话
            for tn in ner_res:
                ans = {}
                ans["label_type"] = tn[1]
                ans['overlap'] = "T" + str(index)

                ans["start_pos"] = text.find(tn[0], posit)
                ans["end_pos"] = ans["start_pos"] + len(tn[0]) - 1
                posit = ans["end_pos"]
                ans["res"] = tn[0]
                item_ner.append(ans)
                index += 1
        test_ner.append(item_ner)
    return test_ner


test_files = os.listdir("./test")
ids = []
starts = []
ends = []
labels = []
ress = []
for file in test_files:
    if not file.endswith(".txt"):
        continue
    id_ = file.split('.')[0]
    with codecs.open("./test/" + file, "r", encoding="utf-8") as f:
        line = f.readlines()
        aa = test_predict(line, NER)
        for line in aa[0]:
            ids.append(id_)
            labels.append(line['label_type'])
            starts.append(line['start_pos'])
            ends.append(line['end_pos'])
            ress.append(line['res'])

df = pd.DataFrame({"ID": ids, "Category": labels, "Pos_b": starts, "Pos_e": ends, "Privacy": ress})
df.to_csv("predict.csv", encoding="utf-8-sig", sep=',', index=False)