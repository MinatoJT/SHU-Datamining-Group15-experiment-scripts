import torch.nn as nn
from transformers import AutoModel  # 改用AutoModel适配Qwen
import torch


class SentimentClassifier(nn.Module):
    def __init__(self, model_name, num_classes):
        super(SentimentClassifier, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        # Qwen2.5-0.5B的隐藏层维度是896
        self.classifier = nn.Linear(896, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # 使用最后一层隐藏状态的均值作为特征
        pooled_output = outputs.last_hidden_state.mean(dim=1)
        output = self.dropout(pooled_output)
        return self.classifier(output)

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))