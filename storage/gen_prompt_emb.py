import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2Model
import os
import matplotlib.pyplot as plt
import seaborn as sns


class GenPromptEmb(nn.Module):
    def __init__(
        self,
        data_path = 'FRED',
        model_name = "gpt2",
        device = 'cuda:0',
        input_len = 96,
        d_model = 768,
        layer = 12,
        divide = 'train'
    ):  
        super(GenPromptEmb, self).__init__()
        self.data_path = data_path
        self.device = device
        self.input_len =  input_len
        self.model_name = model_name
        self.d_model = d_model
        self.layer = layer
        self.len = self.input_len-1

        self.divide = divide
        self.save_freq = 500  # 多少个batch写一次
        self._log_counter = 0
        
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2Model.from_pretrained(model_name).to(self.device)

        # 修改2
        # 定义目标层编号（GPT2共有12层，hidden_states[0]是embedding）
        self.target_layers = [1, 3, 7, 12]  
        # 动态门控网络：输入 pooled 向量（[B, 768]），输出门控权重（[B, 4]）
        self.gate_predictor = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.d_model // 2, len(self.target_layers))  # 输出4个门控分数
        )


    def _prepare_prompt(self, input_template, in_data, in_data_mark, i, j):
        values = in_data[i, :, j].flatten().tolist()
        values_str = ", ".join([str(int(value)) for value in values])

        trends = torch.sum(torch.diff(in_data[i, :, j].flatten()))
        trends_str = f"{trends.item():0f}"
        
        if self.data_path in ['FRED', 'ILI']:
            start_date = f"{int(in_data_mark[i,0,2]):02d}/{int(in_data_mark[i,0,1]):02d}/{int(in_data_mark[i,0,0]):04d}"
            end_date = f"{int(in_data_mark[i,self.len,2]):02d}/{int(in_data_mark[i,self.len,1]):02d}/{int(in_data_mark[i,self.len,0]):04d}"
        elif self.data_path in ['ETTh1', 'ETTh2', 'ECL']:
            start_date = f"{int(in_data_mark[i,0,2]):02d}/{int(in_data_mark[i,0,1]):02d}/{int(in_data_mark[i,0,0]):04d} {int(in_data_mark[i,0,4]):02d}:00"
            end_date = f"{int(in_data_mark[i,self.len,2]):02d}/{int(in_data_mark[i,self.len,1]):02d}/{int(in_data_mark[i,self.len,0]):04d} {int(in_data_mark[i,self.len,4]):02d}:00"
        else:
            start_date = f"{int(in_data_mark[i,0,2]):02d}/{int(in_data_mark[i,0,1]):02d}/{int(in_data_mark[i,0,0]):04d} {int(in_data_mark[i,0,4]):02d}:{int(in_data_mark[i,0,5]):02d}"
            end_date = f"{int(in_data_mark[i,self.len,2]):02d}/{int(in_data_mark[i,self.len,1]):02d}/{int(in_data_mark[i,self.len,0]):04d} {int(in_data_mark[i,self.len,4]):02d}:{int(in_data_mark[i,self.len,5]):02d}"

        in_prompt = input_template.replace("value1, ..., valuen", values_str)
        in_prompt = in_prompt.replace("Trends", trends_str)
        in_prompt = in_prompt.replace("[t1]", start_date).replace("[t2]", end_date)

        tokenized_prompt = self.tokenizer.encode(in_prompt, return_tensors="pt").to(self.device)
        return tokenized_prompt

    def forward(self, tokenized_prompt):
        with torch.no_grad():
            # 原(消融3)
            # outputs = self.model(tokenized_prompt, output_attentions=True)
            # prompt_embeddings = outputs.last_hidden_state

            # # 修改1
            # outputs = self.model(tokenized_prompt, output_attentions=True, output_hidden_states=True)
            # prompt_embeddings = outputs.hidden_states[7]  # 只用一层的输出
            # attentions = outputs.attentions
            # self._log_counter += 1

            # 修改2 
            outputs = self.model(tokenized_prompt, output_attentions=True, output_hidden_states=True)
            hidden_states = outputs.hidden_states  # List[13], 每层 [B, T, H]
            attentions = outputs.attentions
            self._log_counter += 1

            # Step 1: 获取目标层输出 [B, T, H]
            selected_layers = [hidden_states[i] for i in self.target_layers]
            # Step 2: 取 embedding 层（hidden_states[0]）的平均池化表示 [B, H]
            pooled_input = hidden_states[0].mean(dim=1)
            # Step 3: 通过门控网络生成门控权重 [B, 4]
            gate_logits = self.gate_predictor(pooled_input)
            gate_weights = torch.softmax(gate_logits, dim=-1)  # [B, 4]
            # Step 4: 叠加所有目标层输出 → [4, B, T, H]
            stacked_layers = torch.stack(selected_layers, dim=0)
            # Step 5: 变换维度为 [B, 4, T, H]
            stacked_layers = stacked_layers.permute(1, 0, 2, 3)
            # Step 6: 门控权重扩展为 [B, 4, 1, 1]
            gate_weights = gate_weights.unsqueeze(-1).unsqueeze(-1)
            # Step 7: 加权融合层输出 → [B, T, H]
            prompt_embeddings = torch.sum(gate_weights * stacked_layers, dim=1)

            return prompt_embeddings


    def generate_embeddings(self, in_data, in_data_mark):
        input_templates = {
            'FRED': "From [t1] to [t2], the values were value1, ..., valuen every month. The total trend value was Trends",
            'ILI': "From [t1] to [t2], the values were value1, ..., valuen every week. The total trend value was Trends",
            'ETTh1': "From [t1] to [t2], the values were value1, ..., valuen every hour. The total trend value was Trends",
            'ETTh2': "From [t1] to [t2], the values were value1, ..., valuen every hour. The total trend value was Trends",
            'ECL': "From [t1] to [t2], the values were value1, ..., valuen every hour. The total trend value was Trends",
            'ETTm1': "From [t1] to [t2], the values were value1, ..., valuen every 15 minutes. The total trend value was Trends",
            'ETTm2': "From [t1] to [t2], the values were value1, ..., valuen every 15 minutes. The total trend value was Trends",
            'Weather': "From [t1] to [t2], the values were value1, ..., valuen every 10 minutes. The total trend value was Trends"
        }

        input_template = input_templates.get(self.data_path, input_templates['FRED'])
        
        tokenized_prompts = []
        max_token_count = 0
        for i in range(len(in_data)):
            for j in range(in_data.shape[2]):
                tokenized_prompt = self._prepare_prompt(input_template, in_data, in_data_mark, i, j).to(self.device)
                max_token_count = max(max_token_count, tokenized_prompt.shape[1])
                tokenized_prompts.append((i, tokenized_prompt.to(self.device), j))

        in_prompt_emb = torch.zeros((len(in_data), max_token_count, self.d_model, in_data.shape[2]), dtype=torch.float32, device=self.device)

        for i, tokenized_prompt, j in tokenized_prompts:
            prompt_embeddings = self.forward(tokenized_prompt)
            padding_length = max_token_count - tokenized_prompt.shape[1]
            if padding_length > 0:
                last_token_embedding = prompt_embeddings[:, -1, :].unsqueeze(1)
                padding = last_token_embedding.repeat(1, padding_length, 1)
                prompt_embeddings_padded = torch.cat([prompt_embeddings, padding], dim=1)
            else:
                prompt_embeddings_padded = prompt_embeddings
                    
            in_prompt_emb[i, :max_token_count, :, j] = prompt_embeddings_padded
            last_token_emb = in_prompt_emb[:, max_token_count-1:max_token_count, :, :]
            last_token_emb = last_token_emb.squeeze()

        return last_token_emb
