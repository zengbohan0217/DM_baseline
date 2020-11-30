import glob
import numpy as np
from sklearn.model_selection import train_test_split, KFold
import os
import pandas as pd
import codecs


def _cut(sentence):
    """
    将一段文本切分成多个句子
    :param sentence:
    :return:
    """
    new_sentence = []
    sen = []
    for i in sentence:
        if i in ['。', '！', '？', '?'] and len(sen) != 0:
            sen.append(i)
            new_sentence.append("".join(sen))
            sen = []
            continue
        sen.append(i)

    if len(new_sentence) <= 1:  # 一句话超过max_seq_length且没有句号的，用","分割，再长的不考虑了。
        new_sentence = []
        sen = []
        for i in sentence:
            if i.split(' ')[0] in ['，', ','] and len(sen) != 0:
                sen.append(i)
                new_sentence.append("".join(sen))
                sen = []
                continue
            sen.append(i)
    if len(sen) > 0:  # 若最后一句话无结尾标点，则加入这句话
        new_sentence.append("".join(sen))
    return new_sentence


def cut_test_set(text_list, len_treshold):
    cut_text_list = []
    cut_index_list = []
    for text in text_list:

        temp_cut_text_list = []
        text_agg = ''
        if len(text) < len_treshold:
            temp_cut_text_list.append(text)
        else:
            sentence_list = _cut(text)  # 一条数据被切分成多句话
            for sentence in sentence_list:
                if len(text_agg) + len(sentence) < len_treshold:
                    text_agg += sentence
                else:
                    temp_cut_text_list.append(text_agg)
                    text_agg = sentence
            temp_cut_text_list.append(text_agg)  # 加上最后一个句子

        cut_index_list.append(len(temp_cut_text_list))
        cut_text_list += temp_cut_text_list

    return cut_text_list, cut_index_list


def process_one(text_file, lable_file, w_path_, text_length):
    with open(text_file, "r", encoding="UTF-8") as f:
        text = f.read()
    lines, line_len = cut_test_set([text], text_length)
    df = pd.read_csv(lable_file, sep=",", encoding="utf-8")
    q_dic = dict()
    for index, row in df.iterrows():
        cls = row[1]
        start_index = row[2]
        end_index = row[3]
        length = end_index - start_index + 1
        for r in range(length):
            if r == 0:
                q_dic[start_index] = ("B-%s" % cls)
            else:
                q_dic[start_index + r] = ("I-%s" % cls)
    i = 0
    for idx, line in enumerate(lines):
        with codecs.open(w_path_, "a+", encoding="utf-8") as w:
            for str_ in line:
                if str_ == " " or str_ == "" or str_ == "\n" or str_ == "\r":
                    pass
                else:
                    if i in q_dic:
                        tag = q_dic[i]
                    else:
                        tag = "O"  # 大写字母O
                    w.write('%s %s\n' % (str_, tag))
                i += 1
            w.write('\n')


if __name__ == "__main__":
    file_list = glob.glob('./train (1)/data/*.txt')
    kf = KFold(n_splits=5, shuffle=True, random_state=999).split(file_list)
    file_list = np.array(file_list)
    # 设置样本长度
    text_length = 250
    for i, (train_fold, test_fold) in enumerate(kf):
        print(len(file_list[train_fold]), len(file_list[test_fold]))
        train_filelist = list(file_list[train_fold])
        val_filelist = list(file_list[test_fold])
        # train_file
        train_w_path = f'./train (1)/deal_data/train_{i}.txt'
        for file in train_filelist:
            if not file.endswith('.txt'):
                continue
            file_name = file.split(os.sep)[-1].split('.')[0]
            label_file = os.path.join("./train (1)/label", "%s.csv" % file_name)
            process_one(file, label_file, train_w_path, text_length)
        # val_file
        val_w_path = f'./train (1)/deal_val/val_{i}.txt'
        for file in val_filelist:
            if not file.endswith('.txt'):
                continue
            file_name = file.split(os.sep)[-1].split('.')[0]
            label_file = os.path.join("./train (1)/label", "%s.csv" % file_name)
            process_one(file, label_file, val_w_path, text_length)
