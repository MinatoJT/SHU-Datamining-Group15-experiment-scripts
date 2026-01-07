import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

# 新增：抽样、进度条和可视化所需库
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer
from config import Config
from dataset import SentimentDataset
from load_data import DataLoader as DataLoaderClass
from model import SentimentClassifier
import torch.nn as nn
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def set_hf_mirrors():
    """设置Hugging Face镜像，加速模型下载"""
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    os.environ['HF_HOME'] = './hf_cache'


set_hf_mirrors()


def evaluate(model, eval_loader, device):
    """评估模型性能（添加验证集进度条）"""
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    # 验证集进度条
    eval_bar = tqdm(eval_loader, desc="Evaluating", leave=False)

    with torch.no_grad():
        for batch in eval_bar:  # 用进度条迭代
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            # 修正：补全右括号，调整换行格式
            loss = nn.CrossEntropyLoss()(outputs, labels)  # 这里补上右括号
            _, predictions = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(predictions == labels)
            total_predictions += len(labels)
            total_loss += loss.item()

            # 进度条显示当前批次损失
            eval_bar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

    return total_loss / len(eval_loader), correct_predictions.double() / total_predictions

    return total_loss / len(eval_loader), correct_predictions.double() / total_predictions


# 新增：训练后结果可视化函数
def visualize_results(train_losses, val_losses, val_accs, epochs):
    plt.figure(figsize=(12, 5))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), val_accs, label='Validation Accuracy', marker='^', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()

    plt.tight_layout()
    plt.show()


def train(train_texts, train_labels, val_texts=None, val_labels=None):
    """训练模型（添加抽样、进度条和指标记录）"""
    # 新增：训练集20%抽样（不修改后续训练逻辑）
    sample_ratio = 0.2
    sample_size = int(len(train_texts) * sample_ratio)
    if sample_size > 0:
        # 随机抽样，保证文本和标签对应
        random_indices = random.sample(range(len(train_texts)), sample_size)
        train_texts = [train_texts[i] for i in random_indices]
        train_labels = [train_labels[i] for i in random_indices]
        print(f"已抽样20%训练数据：{len(train_texts)}条")

    # 清理GPU缓存（原逻辑不变）
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 加载配置（原逻辑不变）
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained(config.model_name)
    model = SentimentClassifier(config.model_name, config.num_classes)
    model.to(device)

    # 准备训练数据（原逻辑不变）
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, config.max_seq_length)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8, pin_memory=True,
                              persistent_workers=True)

    # 准备验证数据（原逻辑不变）
    if val_texts is not None and val_labels is not None:
        val_dataset = SentimentDataset(val_texts, val_labels, tokenizer, config.max_seq_length)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=8, pin_memory=True)

    # 优化器（原逻辑不变）
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)

    # 新增：记录指标用于可视化
    train_losses = []
    val_losses = []
    val_accs = []

    # 训练循环（添加进度条，原训练逻辑不变）
    best_accuracy = 0
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0

        # 新增：训练进度条
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.num_epochs}")

        for batch in train_bar:  # 用进度条迭代
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = nn.CrossEntropyLoss()(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            # 进度条显示当前批次损失
            train_bar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f'Epoch {epoch + 1}/{config.num_epochs}')
        print(f'Average training loss: {avg_train_loss:.4f}')

        if val_texts is not None and val_labels is not None:
            val_loss, val_accuracy = evaluate(model, val_loader, device)
            val_losses.append(val_loss)
            val_accs.append(val_accuracy.item())
            print(f'Validation Loss: {val_loss:.4f}')
            print(f'Validation Accuracy: {val_accuracy:.4f}')

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                # 新增：创建保存目录（如果不存在）
                import os
                save_dir = os.path.dirname(config.model_save_path)  # 获取模型保存路径的文件夹部分
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)  # 递归创建文件夹
                # 保存模型
                model.save_model(config.model_save_path)

    # 新增：训练完成后可视化
    visualize_results(train_losses, val_losses, val_accs, config.num_epochs)

    return model


if __name__ == "__main__":
    set_hf_mirrors()
    config = Config()
    data_loader = DataLoaderClass(config)

    # 加载数据（原逻辑不变）
    train_texts, train_labels = data_loader.load_csv("train_part_1.csv")
    val_texts, val_labels = data_loader.load_csv("dev.csv")
    test_texts, test_labels = data_loader.load_csv("test.csv")

    # 训练模型
    train(train_texts, train_labels, val_texts, val_labels)