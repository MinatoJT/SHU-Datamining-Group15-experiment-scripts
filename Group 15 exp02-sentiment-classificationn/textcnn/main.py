import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import re
import nltk
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import os

# 找到代码中“下载nltk依赖”的部分，修改为以下代码
try:
    # 测试是否已拥有所有必需资源
    word_tokenize("test")
    stopwords.words('english')
except LookupError:
    # 同时下载punkt_tab（新版分词资源）和stopwords
    nltk.download('punkt_tab')
    nltk.download('stopwords')


# 配置类（统一管理参数，适配你的数据集路径）
class Config:
    def __init__(self):
        # 数据参数（关键：改为当前目录下的dataset文件夹）
        self.data_dir = "dataset"  # 直接写"dataset"即可（因为main.py和dataset在同一文件夹）
        self.train_file = "train_part_1.csv"
        self.dev_file = "dev.csv"
        self.test_file = "test.csv"
        self.max_vocab_size = 500000  # 词汇表最大大小
        self.max_seq_length = 128  # 文本最大长度
        self.embedding_dim = 128  # 词嵌入维度

        # 模型参数
        self.num_filters = 128  # 卷积核数量
        self.filter_sizes = [3, 5]  # 卷积核窗口大小（实验标准）
        self.dropout = 0.5  # dropout概率
        self.num_classes = 2  # 情感分类（1=负面→0，2=正面→1）

        # 训练参数
        self.batch_size = 64
        self.learning_rate = 1e-3
        self.num_epochs = 2
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_save_path = "textcnn_best_model.pth"  # 最优模型保存路径


# 数据预处理工具类（文本清洗、分词、词汇表构建）
class TextPreprocessor:
    def __init__(self, vocab_size=50000):
        self.vocab_size = vocab_size
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}  # 填充符（0）和未知词（1）
        self.stop_words = set(stopwords.words('english'))  # 英文停用词（如the/a/an）

    def clean_text(self, text):
        """文本清洗：去除特殊字符、小写化、分词、去停用词"""
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())  # 只保留字母和空格，转为小写
        tokens = word_tokenize(text)  # 分词（如 "Great CD" → ["great", "cd"]）
        tokens = [token for token in tokens if token not in self.stop_words]  # 去除停用词
        return tokens

    def build_vocab(self, texts, sample_ratio=0.05):  # 新增sample_ratio参数，默认用10%样本
        """用训练集的部分样本构建词汇表（避免内存溢出）"""
        # 采样10%的训练集（180万→18万，足够构建高质量词汇表）
        sample_size = int(len(texts) * sample_ratio)
        sampled_texts = texts[:sample_size]  # 取前10%样本（也可随机采样）

        all_tokens = []
        for text in sampled_texts:
            all_tokens.extend(self.clean_text(text))
        # 统计词频，取前 vocab_size-2 个词（预留PAD和UNK）
        word_freq = Counter(all_tokens).most_common(self.vocab_size - 2)
        for word, _ in word_freq:
            self.word2idx[word] = len(self.word2idx)

    def text_to_ids(self, text):
        """将文本转为索引序列（适配模型输入）"""
        tokens = self.clean_text(text)
        # 词→索引：存在于词汇表则用对应索引，否则用UNK（1）
        ids = [self.word2idx.get(token, self.word2idx["<UNK>"]) for token in tokens]
        return ids


# 数据集类（适配无表头CSV的3列结构）
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, preprocessor, max_seq_length):
        self.texts = texts  # 合并后的文本（短文本+长文本）
        self.labels = labels - 1  # 标签转换：1→0（负面），2→1（正面）（适配CrossEntropyLoss）
        self.preprocessor = preprocessor
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # 文本转索引序列
        text_ids = self.preprocessor.text_to_ids(self.texts[idx])
        # 截断/填充到固定长度（模型要求输入长度一致）
        if len(text_ids) > self.max_seq_length:
            text_ids = text_ids[:self.max_seq_length]  # 过长则截断
        else:
            text_ids += [self.preprocessor.word2idx["<PAD>"]] * (self.max_seq_length - len(text_ids))  # 过短则填充

        return {
            "input_ids": torch.tensor(text_ids, dtype=torch.long),  # 模型输入：文本索引
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)  # 标签
        }


# TextCNN模型（Kim Yoon 2015 标准架构，实验要求）
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, filter_sizes, num_classes, dropout):
        super(TextCNN, self).__init__()
        # 词嵌入层：将词索引转为向量（可后续替换为word2vec预训练向量）
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # 多尺度卷积层：不同窗口大小的卷积核捕捉不同长度的短语特征
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embedding_dim)) for fs in filter_sizes
        ])

        # 全连接层：整合多尺度卷积特征，输出分类结果
        self.dropout = nn.Dropout(dropout)  # 防止过拟合
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)

    def forward(self, input_ids):
        # input_ids: [batch_size, max_seq_length] → 批量文本索引
        embedded = self.embedding(input_ids).unsqueeze(1)  # [batch_size, 1, max_seq_length, embedding_dim] → 增加通道维度

        # 卷积 + 最大池化（提取每个卷积核的最优特征）
        conv_outputs = []
        for conv in self.convs:
            conv_out = conv(embedded)  # [batch_size, num_filters, max_seq_length - fs + 1, 1] → 卷积特征
            conv_out = conv_out.squeeze(-1)  # [batch_size, num_filters, max_seq_length - fs + 1] → 移除多余维度
            pool_out = nn.functional.max_pool1d(conv_out, conv_out.size(2)).squeeze(
                2)  # [batch_size, num_filters] → 最大池化
            conv_outputs.append(pool_out)

        # 拼接多尺度特征 + 分类
        cat_outputs = torch.cat(conv_outputs, dim=1)  # [batch_size, len(filter_sizes)*num_filters] → 特征拼接
        dropout_output = self.dropout(cat_outputs)  # dropout层
        logits = self.fc(dropout_output)  # [batch_size, num_classes] → 分类输出（未归一化的概率）
        return logits


# 评估函数（计算实验要求的所有指标：准确率、F1、AUC-ROC、混淆矩阵）
def evaluate(model, dataloader, device):
    model.eval()  # 模型切换为评估模式（关闭dropout）
    all_preds = []  # 所有预测结果
    all_labels = []  # 所有真实标签
    all_logits = []  # 所有正类输出（用于AUC-ROC计算）

    with torch.no_grad():  # 关闭梯度计算，加速并节省显存
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids)  # 模型预测
            preds = torch.argmax(logits, dim=1)  # 取概率最大的类别作为预测结果

            # 收集结果（转为CPU numpy格式，方便后续计算指标）
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_logits.extend(logits.cpu().numpy()[:, 1])  # 正类（标签1）的输出值

    # 计算评估指标
    accuracy = accuracy_score(all_labels, all_preds)  # 准确率
    f1 = f1_score(all_labels, all_preds, average="weighted")  # 加权F1（适配类别不平衡）
    auc_roc = roc_auc_score(all_labels, all_logits)  # AUC-ROC（衡量分类器区分能力）
    conf_matrix = confusion_matrix(all_labels, all_preds)  # 混淆矩阵（TP/TN/FP/FN）

    return {
        "accuracy": accuracy,
        "f1_score": f1,
        "auc_roc": auc_roc,
        "confusion_matrix": conf_matrix
    }


# 加载数据（关键：按无表头CSV的3列结构读取）
def load_data(config):
    def load_csv(file_path):
        # 读取无表头CSV：header=None → 不将第一行视为表头；usecols=[0,1,2] → 只读取前3列
        df = pd.read_csv(os.path.join(config.data_dir, file_path), header=None, usecols=[0, 1, 2])
        # 列分配：第0列=标签，第1列=短文本，第2列=长文本
        labels = df.iloc[:, 0].values  # 标签（1/2）
        short_texts = df.iloc[:, 1].fillna("")  # 短文本（空值用空字符串填充）
        long_texts = df.iloc[:, 2].fillna("")  # 长文本（空值用空字符串填充）
        # 合并短文本和长文本（包含更完整的情感信息，提升模型效果）
        texts = short_texts + " " + long_texts
        return texts, labels

    # 分别加载训练集、开发集、测试集
    train_texts, train_labels = load_csv(config.train_file)
    dev_texts, dev_labels = load_csv(config.dev_file)
    test_texts, test_labels = load_csv(config.test_file)

    print(f"数据加载完成：")
    print(f"训练集：{len(train_texts)}条样本")
    print(f"开发集：{len(dev_texts)}条样本")
    print(f"测试集：{len(test_texts)}条样本")
    print(f"标签分布（训练集）：负面（1）{sum(train_labels == 1)}条，正面（2）{sum(train_labels == 2)}条")

    return train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels


# 训练主函数（完整流程：数据加载→预处理→模型训练→评估→保存）
def train_textcnn():
    # 1. 初始化配置
    config = Config()
    print(f"\n使用设备：{config.device}")
    # 检查数据目录是否存在（避免路径错误）
    if not os.path.exists(config.data_dir):
        raise FileNotFoundError(f"数据目录不存在：{config.data_dir}，请在Config类中修改正确路径！")

    # 2. 加载数据（按无表头3列结构）
    train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels = load_data(config)

    # 3. 文本预处理：构建词汇表（仅用训练集，避免数据泄露）
    preprocessor = TextPreprocessor(vocab_size=config.max_vocab_size)
    preprocessor.build_vocab(train_texts)
    vocab_size = len(preprocessor.word2idx)
    print(f"\n词汇表构建完成：共{vocab_size}个词（包含PAD和UNK）")

    # 4. 构建数据集和数据加载器（批量加载数据，适配GPU）
    train_dataset = SentimentDataset(train_texts, train_labels, preprocessor, config.max_seq_length)
    dev_dataset = SentimentDataset(dev_texts, dev_labels, preprocessor, config.max_seq_length)
    test_dataset = SentimentDataset(test_texts, test_labels, preprocessor, config.max_seq_length)

    # 数据加载器：shuffle=True（训练集打乱，提升泛化能力）；batch_size=配置值
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

    # 5. 初始化模型、优化器、损失函数
    model = TextCNN(
        vocab_size=vocab_size,
        embedding_dim=config.embedding_dim,
        num_filters=config.num_filters,
        filter_sizes=config.filter_sizes,
        num_classes=config.num_classes,
        dropout=config.dropout
    ).to(config.device)  # 模型移动到GPU/CPU

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)  # Adam优化器（常用且稳定）
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失（分类任务标准损失）

    # 6. 训练循环（按epoch迭代）
    best_dev_f1 = 0.0  # 记录开发集最佳F1（用于保存最优模型）
    print(f"\n开始训练（共{config.num_epochs}个epoch）：")

    for epoch in range(config.num_epochs):
        model.train()  # 模型切换为训练模式（开启dropout）
        total_loss = 0.0  # 记录当前epoch的总损失

        # 批量训练
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(config.device)
            labels = batch["labels"].to(config.device)

            # 梯度清零（避免累计）
            optimizer.zero_grad()
            # 模型预测
            logits = model(input_ids)
            # 计算损失
            loss = criterion(logits, labels)
            # 反向传播（计算梯度）
            loss.backward()
            # 优化器更新参数
            optimizer.step()

            # 累计损失
            total_loss += loss.item()

            # 打印批量训练进度（每10个batch打印一次）
            if (batch_idx + 1) % 10 == 0:
                avg_batch_loss = total_loss / (batch_idx + 1)
                print(
                    f"Epoch {epoch + 1}/{config.num_epochs} | Batch {batch_idx + 1}/{len(train_loader)} | Batch Loss: {avg_batch_loss:.4f}")

        # 计算当前epoch的训练集平均损失
        avg_train_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch + 1}/{config.num_epochs} 训练完成")
        print(f"训练集平均损失：{avg_train_loss:.4f}")

        # 评估开发集（监控模型泛化能力，避免过拟合）
        dev_metrics = evaluate(model, dev_loader, config.device)
        print(f"开发集指标：")
        print(f"  准确率：{dev_metrics['accuracy']:.4f}")
        print(f"  加权F1：{dev_metrics['f1_score']:.4f}")
        print(f"  AUC-ROC：{dev_metrics['auc_roc']:.4f}")
        print(f"  混淆矩阵：")
        print(dev_metrics['confusion_matrix'])

        # 保存最优模型（仅当开发集F1超过历史最佳时保存，避免保存过拟合模型）
        if dev_metrics["f1_score"] > best_dev_f1:
            best_dev_f1 = dev_metrics["f1_score"]
            torch.save(model.state_dict(), config.model_save_path)
            print(f"✅ 开发集F1提升至{best_dev_f1:.4f}，保存最优模型到 {config.model_save_path}")
        else:
            print(f"❌ 开发集F1未提升（当前最佳：{best_dev_f1:.4f}）")

    # 7. 测试集最终评估（用最优模型评估，得到最终结果）
    print(f"\n" + "=" * 50)
    print("=== 测试集最终评估（使用最优模型） ===")
    # 加载最优模型
    model.load_state_dict(torch.load(config.model_save_path))
    test_metrics = evaluate(model, test_loader, config.device)
    # 打印测试集指标
    print(f"测试集准确率：{test_metrics['accuracy']:.4f}")
    print(f"测试集加权F1：{test_metrics['f1_score']:.4f}")
    print(f"测试集AUC-ROC：{test_metrics['auc_roc']:.4f}")
    print(f"测试集混淆矩阵：")
    print(test_metrics['confusion_matrix'])
    print("=" * 50)


# 主函数入口（标准Python工程化写法）
if __name__ == "__main__":
    train_textcnn()