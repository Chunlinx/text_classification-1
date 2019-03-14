#coding:utf-8
"""
对"运动"类增量聚类结果中的每个簇抽取关键词
方法为tf-idf法
"""

import codecs
from pyhanlp import *
import re
from nltk.probability import FreqDist
from nltk.text import TextCollection
import time


NotionalTokenizer = JClass("com.hankcs.hanlp.tokenizer.NotionalTokenizer")


# 仅保留中文字符
def translate(text):
    p2 = re.compile(u'[^\u4e00-\u9fa5]')  # 中文的编码范围是：\u4e00到\u9fa5
    zh = " ".join(p2.split(text)).strip()
    zh = ",".join(zh.split())
    res_str = zh  # 经过相关处理后得到中文的文本
    return res_str


# 预处理，实词分词器分词，查询词语向量，并返回文本向量
def preprocess(text):
    # 去掉非中文字符
    text = translate(text)
    # 将\r\n替换为空格
    text = re.sub(u'[\r\n]+', u' ', text)
    # 分词与词性标注，使用实词分词器
    word_li = NotionalTokenizer.segment(text)
    word_li = [w.word for w in word_li]
    # 去掉单字词
    word_li = [w for w in word_li if len(w)>1]

    return word_li

text_dict = dict()
with codecs.open('data/res_single_pass.txt', 'rb', 'utf-8', 'ignore') as infile:
    for line in infile:
        line = line.strip()
        if line:
            cluster_ser, text = line.split(u'\t')
            text_dict.setdefault(cluster_ser, [])
            text_dict[cluster_ser].append(text)

outfile = open('data/cluster_keywords.txt', 'wb')
outfile2 = open('data/cluster_keywords2.txt', 'wb')

for cluster_ser, text_li in text_dict.items():
    print("cluster", cluster_ser, "text cnt=", len(text_li))
    # if cluster_ser == "3" or cluster_ser == "5":
    #     continue
    t0 = time.time()
    vocabulary_set = set()
    cluster_text_li = []
    for text_ser, text in enumerate(text_li):
        word_li = preprocess(text)
        cluster_text_li.append(tuple(word_li))
        vocabulary_set |= set(word_li)
    t1 = time.time()
    print("预处理簇内文本 %.2f s, 词汇表长度 = %d" % ((t1-t0), len(vocabulary_set)) )
    stats = TextCollection(cluster_text_li)
    fdist = FreqDist([w for text in cluster_text_li for w in text])
    t0 = time.time()
    word_li = []
    for word in vocabulary_set:
        # 计算词语在簇内的tf值
        word_tf = fdist.freq(word)
        # 计算词语在簇内文档间的idf值
        word_idf = stats.idf(word)
        # 计算词语tf-idf值
        word_tf_idf = word_tf * word_idf
        if len(cluster_text_li) > 1:
            word_li.append((word, word_tf_idf))
        else:
            word_li.append((word, word_tf))
    t1 = time.time()
    print("计算词语tf值idf值tf_idf值 %.2f s" % (t1 - t0))
    word_li = sorted(word_li, key=lambda x:x[1], reverse=True)

    out_str = u'%s\t%s\n' %(cluster_ser, u' '.join([u'%s:%.3f' % (w[0],w[1]) for w in word_li[:10]]))
    outfile.write(out_str.encode('utf-8', 'ignore'))

    out_str = u'%s\t%s\n' %(cluster_ser, u' '.join([u'%s:%.3f' % (w[0],w[1]) for w in word_li[-10:]]))
    outfile2.write(out_str.encode('utf-8', 'ignore'))

outfile.close()
outfile2.close()
