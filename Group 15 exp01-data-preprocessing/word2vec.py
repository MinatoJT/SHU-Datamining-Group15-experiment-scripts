import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import re
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx  # 后续node2vec实验可用，先导入
import warnings

warnings.filterwarnings('ignore')


# -------------------------- 1. 文本预处理模块（原有增强） --------------------------
def preprocess_text(text):
    """
    文本预处理函数：处理缺失值、小写转换、移除特殊字符、分词
    :param text: 原始文本字符串
    :return: 分词后的词汇列表
    """
    if pd.isna(text):
        return []
    # 转换为小写
    text = text.lower()
    # 移除特殊字符和数字（可选，根据实验需求调整）
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    # 简单分词（也可替换为nltk.word_tokenize，需下载punkt）
    tokens = text.split()
    # 移除停用词（可选，提升词向量质量）
    stop_words = set(['a', 'an', 'the', 'and', 'or', 'but', 'is', 'are'])
    tokens = [token for token in tokens if token not in stop_words and len(token) > 1]
    return tokens


def load_and_preprocess_data(file_path):
    """
    加载并预处理Amazon评论数据
    :param file_path: 数据文件路径
    :return: 预处理后的语料库、标签、原始DataFrame
    """
    # 读取CSV文件（适配Amazon数据集格式）
    df = pd.read_csv(file_path, on_bad_lines='skip')  # 跳过错误行
    # 确保至少有3列（标签、标题、评论）
    if df.shape[1] < 3:
        raise ValueError("数据集需包含至少3列：标签、标题、评论内容")

    # 合并标题和评论内容，避免类型错误
    df['text'] = df.iloc[:, 1].astype(str) + " " + df.iloc[:, 2].astype(str)

    # 预处理所有文本生成语料库
    corpus = [preprocess_text(text) for text in df['text']]

    return corpus, df.iloc[:, 0].values, df


# -------------------------- 2. Word2Vec模型训练（支持两种模式） --------------------------
def train_word2vec_model(corpus, model_type='cbow'):
    """
    训练Word2Vec模型，支持CBOW和Skip-gram两种模式
    :param corpus: 预处理后的语料库
    :param model_type: 模型类型，'cbow'或'skipgram'
    :return: 训练好的Word2Vec模型
    """
    # sg=0表示CBOW，sg=1表示Skip-gram
    sg = 1 if model_type == 'skipgram' else 0
    model = Word2Vec(
        sentences=corpus,
        vector_size=100,  # 词向量维度
        window=5,  # 上下文窗口大小
        min_count=5,  # 过滤低频词（提升模型质量）
        workers=4,  # 并行线程数
        sg=sg,  # 模型模式
        epochs=10  # 训练轮数
    )
    print(f"✅ {model_type.upper()}模型训练完成，词汇表大小：{len(model.wv)}")
    return model


# -------------------------- 3. Word2Vec架构图生成（可视化原理） --------------------------
def plot_word2vec_architecture(model_type='cbow'):
    """
    绘制Word2Vec两种模式的神经网络架构图（简化版）
    :param model_type: 模型类型，'cbow'或'skipgram'
    """
    plt.figure(figsize=(10, 6))
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文
    plt.title(f"Word2Vec-{model_type.upper()} 神经网络架构", fontsize=14)

    # 定义图层位置
    layers = {
        '输入层': 0.8,
        '隐藏层': 0.5,
        '输出层': 0.2
    }

    # CBOW：输入上下文词→隐藏层→输出目标词
    if model_type == 'cbow':
        # 输入层（多个上下文词）
        for i in range(4):  # 示例：4个上下文词
            plt.scatter(0.2, layers['输入层'] - i * 0.1, s=200, c='lightblue', label='上下文词' if i == 0 else "")
        # 隐藏层（词向量）
        plt.scatter(0.5, layers['隐藏层'], s=300, c='orange', label='隐藏层（词向量）')
        # 输出层（目标词）
        plt.scatter(0.8, layers['输出层'], s=200, c='lightgreen', label='目标词')

    # Skip-gram：输入目标词→隐藏层→输出上下文词
    else:
        # 输入层（目标词）
        plt.scatter(0.2, layers['输入层'], s=200, c='lightblue', label='目标词')
        # 隐藏层（词向量）
        plt.scatter(0.5, layers['隐藏层'], s=300, c='orange', label='隐藏层（词向量）')
        # 输出层（多个上下文词）
        for i in range(4):
            plt.scatter(0.8, layers['输出层'] - i * 0.1, s=200, c='lightgreen', label='上下文词' if i == 0 else "")

    # 绘制连接线
    plt.axvline(x=0.2, ymin=0.1, ymax=0.9, c='gray', linestyle='--')
    plt.axvline(x=0.5, ymin=0.1, ymax=0.9, c='gray', linestyle='--')
    plt.axvline(x=0.8, ymin=0.1, ymax=0.9, c='gray', linestyle='--')

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(loc='upper right')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"word2vec_{model_type}_architecture.png", dpi=300, bbox_inches='tight')
    print(f"📌 Word2Vec-{model_type.upper()}架构图已保存为word2vec_{model_type}_architecture.png")


# -------------------------- 4. 向量相似度计算（实验三要求） --------------------------
def calculate_vector_similarity(model, word1, word2=None, top_n=10):
    """
    计算词向量相似度：单个词的相似词/两个词的余弦相似度
    :param model: 训练好的Word2Vec模型
    :param word1: 目标词1
    :param word2: 目标词2（可选，若为None则返回word1的相似词）
    :param top_n: 返回相似词的数量
    :return: 相似度结果
    """
    if word1 not in model.wv:
        return f"❌ 词汇'{word1}'不在词汇表中"

    # 计算两个词的余弦相似度
    if word2 is not None:
        if word2 not in model.wv:
            return f"❌ 词汇'{word2}'不在词汇表中"
        similarity = cosine_similarity([model.wv[word1]], [model.wv[word2]])[0][0]
        return f"📊 '{word1}'与'{word2}'的余弦相似度：{similarity:.4f}"

    # 返回相似词列表
    similar_words = model.wv.most_similar(word1, topn=top_n)
    result = [f"📈 与'{word1}'最相似的{top_n}个词："]
    for word, score in similar_words:
        result.append(f"   {word}: {score:.4f}")
    return "\n".join(result)


# -------------------------- 5. T-SNE可视化（实验四要求） --------------------------
def tsne_visualization(model, top_n_words=50):
    # 选取高频词
    words = list(model.wv.index_to_key)[:top_n_words]
    vectors = [model.wv[word] for word in words]
    # 将列表转为numpy数组
    vectors = np.array(vectors)  # 这一步是核心！

    # T-SNE降维（2维）
    tsne = TSNE(n_components=2, random_state=42, perplexity=10)
    vectors_tsne = tsne.fit_transform(vectors)

    # 绘制可视化图
    plt.figure(figsize=(12, 8))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.title("Word2Vec词向量T-SNE可视化", fontsize=14)

    for i, word in enumerate(words):
        plt.scatter(vectors_tsne[i, 0], vectors_tsne[i, 1], c='blue', alpha=0.7)
        plt.text(vectors_tsne[i, 0] + 0.1, vectors_tsne[i, 1] + 0.1, word, fontsize=9)

    plt.xlabel("T-SNE维度1")
    plt.ylabel("T-SNE维度2")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig("word2vec_tsne_visualization.png", dpi=300, bbox_inches='tight')
    print("📊 T-SNE可视化图已保存为word2vec_tsne_visualization.png")


# -------------------------- 6. 主函数（串联所有实验步骤） --------------------------
def main():
    # 1. 数据加载与预处理
    try:
        corpus, labels, df = load_and_preprocess_data('train_part_1.csv')
        print(f"📚 数据加载完成，语料库样本数：{len(corpus)}")
    except FileNotFoundError:
        print("❌ 未找到数据文件，请检查文件路径是否正确")
        return

    # 2. 训练CBOW和Skip-gram两种模型
    cbow_model = train_word2vec_model(corpus, model_type='cbow')
    skipgram_model = train_word2vec_model(corpus, model_type='skipgram')

    # 3. 绘制Word2Vec架构图
    plot_word2vec_architecture(model_type='cbow')
    plot_word2vec_architecture(model_type='skipgram')

    # 4. 计算词向量相似度（示例）
    print("\n" + "-" * 50)
    print(calculate_vector_similarity(cbow_model, "great"))
    print(calculate_vector_similarity(cbow_model, "great", "excellent"))

    # 5. T-SNE可视化
    tsne_visualization(cbow_model, top_n_words=50)

    # 6. 保存模型
    cbow_model.save("word2vec_cbow_model.model")
    skipgram_model.save("word2vec_skipgram_model.model")
    print("\n💾 模型已保存，实验一至实验四核心要求完成！")


if __name__ == "__main__":
    main()