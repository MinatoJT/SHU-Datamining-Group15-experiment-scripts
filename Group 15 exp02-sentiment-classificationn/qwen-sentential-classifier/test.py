import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

# 导入你的项目模块
from config import Config
from model import SentimentClassifier
from load_data import DataLoader as DataLoaderClass
from dataset import SentimentDataset
# 导入 evaluate 函数
from main import evaluate, set_hf_mirrors


# ==========================================
# 模块 1: 绘图功能 (数据来自你的日志)
# ==========================================
def plot_reconstructed_history():
    print("\n[1/3] 正在根据历史日志生成训练图表...")

    # 设置中文字体
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Microsoft YaHei"]
    plt.rcParams['axes.unicode_minus'] = False
    sns.set(font='SimHei', font_scale=1.2)

    # === 手动录入的训练数据 ===
    epochs = [1, 2, 3, 4, 5]
    train_losses = [0.3554, 0.1298, 0.0295, 0.0060, 0.0007]
    val_losses = [0.2681, 0.5123, 0.4448, 0.6190, 0.8904]
    val_accs = [0.9251, 0.9331, 0.9421, 0.9461, 0.9441]

    # === 绘图逻辑 ===
    plt.figure(figsize=(12, 5))

    # 子图1: 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-o', label='训练集损失 (Train Loss)')
    plt.plot(epochs, val_losses, 'r-s', label='验证集损失 (Val Loss)')
    plt.title('训练与验证损失曲线')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 子图2: 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accs, 'g-^', label='验证集准确率 (Val Accuracy)')
    plt.title('验证集准确率变化')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    # 标注最高点
    max_acc = max(val_accs)
    max_epoch = epochs[val_accs.index(max_acc)]
    plt.annotate(f'峰值: {max_acc:.4f}',
                 xy=(max_epoch, max_acc),
                 xytext=(max_epoch, max_acc - 0.005),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 保存图片
    save_path = "training_plots_reconstructed.png"
    plt.savefig(save_path, dpi=300)
    print(f"✅ 图表已保存为: {save_path}")

    # 尝试显示（如果在支持GUI的环境下）
    try:
        plt.show(block=False)
        plt.pause(1)
        plt.close()
    except:
        pass


# ==========================================
# 模块 2: 本地预测函数
# ==========================================
def predict_local(text, model, tokenizer, device, config):
    model.eval()
    with torch.no_grad():
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=config.max_seq_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        outputs = model(input_ids, attention_mask)
        _, predictions = torch.max(outputs, dim=1)

    return predictions.item()


# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    # --- 步骤 1: 生成图片 ---
    plot_reconstructed_history()

    # --- 步骤 2: 准备测试环境 ---
    print("\n[2/3] 初始化模型与测试环境...")
    set_hf_mirrors()
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载模型架构
    model = SentimentClassifier(config.model_name, config.num_classes)

    # 加载权重
    if os.path.exists(config.model_save_path):
        print(f"✅ 正在加载保存的模型: {config.model_save_path}")
        state_dict = torch.load(config.model_save_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(f"❌ 错误：未找到模型文件 {config.model_save_path}")
        exit()

    model.to(device)
    model.eval()

    # 准备分词器
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # --- 步骤 3: 运行测试集评估 ---
    print("\n===== 正在加载测试集 =====")
    data_loader = DataLoaderClass(config)
    test_texts, test_labels = data_loader.load_csv(config.test_path)

    test_dataset = SentimentDataset(test_texts, test_labels, tokenizer, config.max_seq_length)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    print("开始测试集评估...")
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"\n📊 测试集最终准确率: {test_acc:.4f}")

    # --- 步骤 4: 运行样例预测 ---
    print("\n[3/3] 运行样例预测")
    examples = [
        "这个产品质量非常好，我很满意！",
        "物流太慢了，包装也破损了，差评。",
        "虽然价格有点贵，但是物有所值。",
        "一般般吧，没有想象中那么好。",
        "The quality is amazing, I love it!",
        "Terrible experience, waste of money."
    ]

    print(f"{'文本':<40} | {'预测结果'}")
    print("-" * 60)
    for text in examples:
        prediction = predict_local(text, model, tokenizer, device, config)
        label = "正面 (Positive)" if prediction == 1 else "负面 (Negative)"
        print(f"{text:<40} | {label}")

    print("\n✅ 所有任务已完成！请查看生成的 'training_plots_reconstructed.png'")