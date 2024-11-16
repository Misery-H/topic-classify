from gutenbergpy import textget

# 中文分词库
import jieba

# 移除停用词
import pandas as pd

# load stopword list
stopwords = pd.read_csv(r'stopwords-master/baidu_stopwords.txt', encoding='utf-8')  # 确保文件编码为utf-8
# 用于词形还原和词性标注
import jieba.posseg as psg

# 用于主题建模，潜在狄利克雷分配（LDA）模型
from gensim import corpora, models, similarities
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel

# LDA评估
import pyLDAvis
import pyLDAvis.gensim_models as gensimvisualize


stopwords = [i[0] for i in stopwords.values]
def preprocess_text(text):
    """
    分词，去除停用词
    """
    words = jieba.cut(text)
    words = [word for word in words if word not in stopwords and len(word) > 1]
    return list(words)
if __name__ == '__main__':

    data = pd.read_csv('data.csv', encoding='utf-8', header=None)
    #读取data的第二列
    data = data.iloc[:, 1]
    #将data的第二列的数据进行预处理
    data['processed_text'] = data.apply(preprocess_text)
    texts = data['processed_text'].tolist()

    # 加载词典
    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=2)

    # 生成语料库
    corpus = [dictionary.doc2bow(text) for text in texts]

    # 训练LDA模型
    lda_model = models.LdaModel(corpus=corpus, id2word=dictionary, random_state=4583, chunksize=20, num_topics=7, passes=200, iterations=400)

    # 打印LDA主题
    for topic in lda_model.print_topics(num_topics=7, num_words=10):
        print(topic)
    coherence_model = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
    coherence_score = coherence_model.get_coherence()
    print(coherence_score)

    dickens_visual = gensimvisualize.prepare(lda_model, corpus, dictionary, mds='mmds')
    pyLDAvis.display(dickens_visual)