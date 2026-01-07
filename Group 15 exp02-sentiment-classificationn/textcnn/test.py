import torch
import os
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# 1. 尝试导入 Config 和 TextCNN
try:
    from main import Config, TextCNN
except ImportError as e:
    print("❌ 导入错误: 无法从 main.py 导入 Config 或 TextCNN。")
    exit()

# ==========================================
# 2. 本地重新实现数据处理 (避免依赖 main.py 函数)
# ==========================================

# 确保 nltk 资源存在
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')


def preprocess_text(text):
    text = str(text).lower()
    return word_tokenize(text)


def build_vocab(texts, max_vocab_size=50000):
    print("正在统计词频...")
    counter = Counter()
    for text in texts:
        tokens = preprocess_text(text)
        counter.update(tokens)

    vocab = {"<PAD>": 0, "<UNK>": 1}
    most_common = counter.most_common(max_vocab_size - 2)
    for word, _ in most_common:
        vocab[word] = len(vocab)
    return vocab


class TextCNNDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        tokens = preprocess_text(text)
        unk_idx = self.vocab.get("<UNK>", 1)
        indices = [self.vocab.get(t, unk_idx) for t in tokens]

        if len(indices) < self.max_len:
            indices += [self.vocab.get("<PAD>", 0)] * (self.max_len - len(indices))
        else:
            indices = indices[:self.max_len]

        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)


def get_data_loader(file_path, vocab, config, shuffle=False):
    df = pd.read_csv(file_path, header=None, names=['label', 'title', 'text'])
    # 标签映射: 假设训练时 1->0, 2->1
    df['label'] = df['label'].map({1: 0, 2: 1})
    texts = (df['title'].fillna("") + " " + df['text'].fillna("")).tolist()
    labels = df['label'].tolist()

    # 获取最大长度，如果 config 没有则默认 128
    max_len = getattr(config, 'max_seq_length', 128)

    dataset = TextCNNDataset(texts, labels, vocab, max_len)
    return DataLoader(dataset, batch_size=getattr(config, 'batch_size', 64), shuffle=shuffle)


# ==========================================
# 3. 主测试逻辑
# ==========================================
def test_model():
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # --- 自动寻找模型文件 ---
    possible_paths = [
        "saved_models/textcnn_best_model.pth",  # 你报错中提到的名字
        "saved_models/sentiment_model.pth",  # 默认名字
        "sentiment_model.pth",
        config.model_save_path if hasattr(config, 'model_save_path') else "none"
    ]

    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            break

    if model_path is None:
        print(f"❌ 错误：在 saved_models/ 下找不到模型文件。")
        print(f"请将你的 .pth 文件重命名为 'sentiment_model.pth' 并放入 saved_models 文件夹。")
        return
    else:
        print(f"🚀 找到模型文件: {model_path}")

    # --- 步骤 A: 重建词汇表 ---
    train_file = os.path.join(getattr(config, 'data_dir', 'dataset'), getattr(config, 'train_file', 'train.csv'))
    # 如果找不到 train.csv，尝试找 train_part_1.csv (根据你之前的config)
    if not os.path.exists(train_file):
        train_file = os.path.join("dataset", "train_part_1.csv")

    if not os.path.exists(train_file):
        print(f"❌ 错误：找不到训练文件 {train_file}，无法重建词汇表。")
        return

    print("正在读取训练集重建词汇表...")
    train_df = pd.read_csv(train_file, header=None, names=['label', 'title', 'text'])
    train_texts = (train_df['title'].fillna("") + " " + train_df['text'].fillna("")).tolist()

    vocab_size = getattr(config, 'max_vocab_size', 50000)
    vocab = build_vocab(train_texts, max_vocab_size=vocab_size)
    print(f"✅ 词汇表重建完成，大小: {len(vocab)}")

    # --- 步骤 B: 初始化模型 (修复参数缺失问题) ---
    print("正在初始化模型...")

    # 这里我们显式传入参数，不再依赖 config 属性
    # 这些是 TextCNN 的经典默认参数，通常你在 main.py 里也是这么写的
    embedding_dim = getattr(config, 'embedding_dim', 100)  # 词向量维度
    num_filters = getattr(config, 'num_filters', 100)  # 卷积核数量
    filter_sizes = getattr(config, 'filter_sizes', [3, 4, 5])  # 卷积核尺寸
    num_classes = 2
    dropout = getattr(config, 'dropout', 0.5)

    try:
        # ✅ 修复点：按顺序传入所有参数
        model = TextCNN(len(vocab), embedding_dim, num_filters, filter_sizes, num_classes, dropout)
    except TypeError as e:
        print(f"⚠️ 初始化尝试1失败: {e}")
        try:
            # 备选：有些实现把 config 放在第一个
            model = TextCNN(config, len(vocab))
        except TypeError as e2:
            print(f"❌ 模型初始化彻底失败。请检查 main.py 中 TextCNN 的 __init__ 定义。")
            print(f"详情: {e2}")
            return

    model.to(device)

    # --- 步骤 C: 加载权重 ---
    print("正在加载权重...")
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print("✅ 权重加载成功！")
    except RuntimeError as e:
        print(f"❌ 权重加载失败 (尺寸不匹配): {e}")
        print("原因：重建的词汇表大小与训练时不同，或者卷积核参数不一致。")
        print("解决：请确保 train_file 指向的文件与训练时完全一致。")
        return

    # --- 步骤 D: 评估 ---
    test_file = os.path.join(getattr(config, 'data_dir', 'dataset'), getattr(config, 'test_file', 'test.csv'))
    if not os.path.exists(test_file):
        print("找不到测试文件，跳过评估")
        return

    print("正在加载测试集...")
    test_loader = get_data_loader(test_file, vocab, config, shuffle=False)

    print("\n===== 开始评估 =====")
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')

    print("\n" + "=" * 30)
    print(f"📊 TextCNN 测试结果")
    print("=" * 30)
    print(f"Accuracy : {acc:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("\n详细报告:")
    print(classification_report(all_labels, all_preds, target_names=['负面', '正面']))


if __name__ == "__main__":
    test_model()