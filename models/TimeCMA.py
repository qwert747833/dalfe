import torch.nn as nn
import torch
from einops import rearrange
from layers.StandardNorm import Normalize
from layers.Cross_Modal_Align import CrossModal
from layers.BiCrossAttention import BiCrossAttention

class Dual(nn.Module):
    def __init__(
        self,
        device = "cuda:7",
        channel = 32,# 128没效果
        num_nodes = 7,
        seq_len = 96,
        pred_len = 96,
        dropout_n = 0.5,# 原本0.1
        d_llm = 768,
        e_layer = 3,# 加深编码器 原本是1 没效果
        d_layer = 2,# 加深解码器 原本是1 没效果
        d_ff=32,
        head =8
    ):
        super().__init__()

        self.device = device
        self.channel = channel
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.dropout_n= dropout_n
        self.d_llm = d_llm
        self.e_layer = e_layer
        self.d_layer = d_layer
        self.d_ff = d_ff
        self.head = head

        self.normalize_layers = Normalize(self.num_nodes, affine=False).to(self.device)
        self.length_to_feature = nn.Linear(self.seq_len, self.channel).to(self.device)

        self.concat_projection = nn.Linear(self.channel + self.d_llm, self.channel).to(self.device)

        # Time Series Encoder
        self.ts_encoder_layer = nn.TransformerEncoderLayer(d_model = self.channel, nhead = self.head, batch_first=True, 
                                                           norm_first = True,dropout = self.dropout_n).to(self.device)
        self.ts_encoder = nn.TransformerEncoder(self.ts_encoder_layer, num_layers = self.e_layer).to(self.device)

        # Prompt Encoder
        self.prompt_encoder_layer = nn.TransformerEncoderLayer(d_model = self.d_llm, nhead = self.head, batch_first=True, 
                                                               norm_first = True,dropout = self.dropout_n).to(self.device)
        self.prompt_encoder = nn.TransformerEncoder(self.prompt_encoder_layer, num_layers = self.e_layer).to(self.device)

        # # change1 Cross-modality alignment
        # self.cross_layer = nn.TransformerDecoderLayer(d_model = self.num_nodes, nhead = 1, batch_first=True, norm_first = True,dropout = self.dropout_n).to(self.device)
        # self.cross = nn.TransformerDecoder(self.cross_layer, num_layers = 1).to(self.device)
        # self.cross = CrossModal(d_model= self.num_nodes, n_heads= 1, d_ff=self.d_ff, norm='LayerNorm', attn_dropout=self.dropout_n, 
        #                         dropout=self.dropout_n, pre_norm=True, activation="gelu", res_attention=True, n_layers=1, store_attn=False).to(self.device)
       
        # # change1 修改 双向交叉注意力
        self.cross = BiCrossAttention(
            d_model_ts=self.channel,
            d_model_emb=self.d_llm,
            nhead=self.head,
            dropout=self.dropout_n
        ).to(self.device)


        # Transformer decoder
        self.decoder_layer = nn.TransformerDecoderLayer(d_model = self.channel, nhead = self.head, batch_first=True, norm_first = True, dropout = self.dropout_n).to(self.device)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers = self.d_layer).to(self.device)

        # Projection
        self.c_to_length = nn.Linear(self.channel, self.pred_len, bias=True).to(self.device)

    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, input_data, input_data_mark, embeddings, return_attn=False):
        input_data = input_data.float()
        input_data_mark = input_data_mark.float()
        embeddings = embeddings.float()

        # RevIN
        input_data = self.normalize_layers(input_data, 'norm')

        input_data = input_data.permute(0,2,1) # [B, N, L]
        input_data = self.length_to_feature(input_data) # [B, N, C]

        embeddings = embeddings.float()
        embeddings = embeddings.squeeze(-1) # [B, E, N]
        embeddings = embeddings.permute(0,2,1) # [B, N, E]

        # Encoderforw
        enc_out = self.ts_encoder(input_data) # [B, N, C]
        enc_out = enc_out.permute(0,2,1) # [B, C, N]
        embeddings = self.prompt_encoder(embeddings) # [B, N, E]
        embeddings = embeddings.permute(0,2,1) # [B, E, N]

        # Cross
        # cross_out = self.cross(enc_out, embeddings, embeddings) # Q X KV  [B, C, N]X[B, E, N] = [B, C, N]
        # change2 修改 原 双向只要两个输入
        if return_attn:
            cross_out, (ts_weights, emb_weights) = self.cross(enc_out, embeddings, return_weights=True)
            return cross_out, (ts_weights, emb_weights)
        else:
            cross_out = self.cross(enc_out, embeddings)

        # cross_out = self.cross(enc_out, embeddings)
        cross_out = cross_out.permute(0,2,1) # [B, N, C]

        # # 消融1 双重对齐改成简单拼接
        # # 首先将两个张量调整到相同的维度顺序 [B, N, C] 和 [B, N, E]
        # enc_out = enc_out.permute(0,2,1) # [B, N, C]
        # embeddings = embeddings.permute(0,2,1) # [B, N, E]

        # # 在特征维度上进行拼接 (C + E)
        # cross_out = torch.cat([enc_out, embeddings], dim=-1) # [B, N, C+E]
        # cross_out = cross_out.to(self.device)

        # # 如果需要保持输出维度与原始一致 [B, N, C]，可以添加一个线性变换
        # cross_out = self.concat_projection(cross_out) # [B, N, C]


        # Decoder
        dec_out = self.decoder(cross_out, cross_out) # [B, N, C]

        # Projection
        dec_out = self.c_to_length(dec_out) # [B, N, L]
        dec_out = dec_out.permute(0,2,1) # [B, L, N]

        # denorm
        dec_out = self.normalize_layers(dec_out, 'denorm')

        return dec_out