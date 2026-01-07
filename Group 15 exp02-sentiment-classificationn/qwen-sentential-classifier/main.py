import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from config import Config
from dataset import SentimentDataset
from load_data import DataLoader as DataLoaderClass
from model import SentimentClassifier
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import random

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
sns.set(font='SimHei', font_scale=1.2)

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def set_hf_mirrors():
    """设置Hugging Face镜像加速下载"""
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    os.environ['HF_HOME'] = './hf_cache'


set_hf_mirrors()


def sample_data(texts, labels, ratio=0.1):
    """从大规模数据集中抽样"""
    assert len(texts) == len(labels), "文本和标签长度不匹配"
    sample_size = int(len(texts) * ratio)
    # 随机抽样（固定种子保证可复现）
    random.seed(42)
    indices = random.sample(range(len(texts)), sample_size)
    sampled_texts = [texts[i] for i in indices]
    sampled_labels = [labels[i] for i in indices]
    print(f"从{len(texts)}条数据中抽样{sample_size}条用于训练")
    return sampled_texts, sampled_labels


def evaluate(model, eval_loader, device):
    """带进度条的评估函数"""
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="评估中", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = nn.CrossEntropyLoss()(outputs, labels)

            _, predictions = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(predictions == labels)
            total_predictions += len(labels)
            total_loss += loss.item()

    return total_loss / len(eval_loader), correct_predictions.double() / total_predictions


def train(train_texts, train_labels, val_texts=None, val_labels=None):
    """训练函数（带可视化和进度条）"""
    # 清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 初始化tokenizer（Qwen专用配置）
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Qwen需要手动设置pad_token

    # 初始化模型
    model = SentimentClassifier(config.model_name, config.num_classes)
    model.to(device)

    # 抽样训练数据
    train_texts, train_labels = sample_data(train_texts, train_labels, config.sample_ratio)

    # 准备数据加载器
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, config.max_seq_length)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    val_loader = None
    if val_texts is not None and val_labels is not None:
        val_dataset = SentimentDataset(val_texts, val_labels, tokenizer, config.max_seq_length)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    # 优化器和调度器
    total_steps = len(train_loader) * config.num_epochs
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    # 记录训练历史用于可视化
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
    best_accuracy = 0

    # 训练循环
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0

        # 训练进度条
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config.num_epochs}', leave=True)
        for batch in train_pbar:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = nn.CrossEntropyLoss()(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            train_pbar.set_postfix({'batch_loss': f'{loss.item():.4f}'})

        # 记录训练损失
        avg_train_loss = total_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        print(f'Average training loss: {avg_train_loss:.4f}')

        # 验证
        if val_loader is not None:
            val_loss, val_accuracy = evaluate(model, val_loader, device)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy.item())

            print(f'Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_accuracy:.4f}')

            # 保存最佳模型
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                os.makedirs(os.path.dirname(config.model_save_path), exist_ok=True)
                model.save_model(config.model_save_path)
                print(f"保存最佳模型 (准确率: {val_accuracy:.4f})")

    # 绘制训练曲线
    plot_training_history(history, config.plot_save_path)
    return model


def plot_training_history(history, save_path):
    """绘制训练可视化曲线"""
    plt.figure(figsize=(12, 5))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='训练损失')
    if history['val_loss']:
        plt.plot(history['val_loss'], label='验证损失')
    plt.title('训练与验证损失')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.legend()

    # 准确率曲线
    plt.subplot(1, 2, 2)
    if history['val_accuracy']:
        plt.plot(history['val_accuracy'], label='验证准确率', color='green')
        plt.title('验证准确率')
        plt.xlabel('轮次')
        plt.ylabel('准确率')
        plt.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"训练可视化结果已保存至: {save_path}")
    plt.close()


def predict(text, model_path=None):
    """预测函数"""
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = SentimentClassifier(config.model_name, config.num_classes)
    if model_path:
        model.load_model(model_path)
    model.to(device)
    model.eval()

    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=config.max_seq_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    with torch.no_grad():
        outputs = model(
            encoding['input_ids'].to(device),
            encoding['attention_mask'].to(device)
        )
        _, predictions = torch.max(outputs, dim=1)

    return predictions.item()


if __name__ == "__main__":
    set_hf_mirrors()
    config = Config()

    # 加载数据
    data_loader = DataLoaderClass(config)
    print("加载训练集...")
    train_texts, train_labels = data_loader.load_csv(config.train_path)
    print("加载验证集...")
    val_texts, val_labels = data_loader.load_csv(config.dev_path)
    print("加载测试集...")
    test_texts, test_labels = data_loader.load_csv(config.test_path)

    # 训练模型
    print("开始训练模型...")
    model = train(train_texts, train_labels, val_texts, val_labels)

    # 预测示例
    print("\n===== 预测示例 =====")
    examples = [
        "这个产品质量非常好，我很满意！",
        "太差劲了，完全不值这个价格，不会再买了",
        "包装很精美，物流也很快，给个好评",
        "东西一般般，没有宣传的那么好，不太推荐",
        "客服态度很好，解决问题很及时，点赞",
        "体验非常差，以后再也不会光顾了"
    ]
    for text in examples:
        pred = predict(text, config.model_save_path)
        print(f"文本: {text}")
        print(f"预测结果: {'正面' if pred == 1 else '负面'}\n")