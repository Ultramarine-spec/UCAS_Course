import math
import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class FeedforwardNN(nn.Module):
    def __init__(self, vocab_size, num_history, embeds_size, hidden_size, dropout):
        super(FeedforwardNN, self).__init__()
        self.embeds_size = embeds_size
        self.num_history = num_history
        self.embedding = nn.Embedding(vocab_size, embeds_size)
        self.linear1 = nn.Linear(num_history * embeds_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):
        input_embeds = self.embedding(input_ids)

        input_embeds = input_embeds.view(-1, self.num_history * self.embeds_size)

        ffn_output = torch.relu(self.linear1(input_embeds))

        ffn_output = self.dropout(ffn_output)

        ffn_output = self.linear2(ffn_output)

        return ffn_output


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embeds_size, hidden_size, num_layers, dropout):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embeds_size)
        self.lstm = nn.LSTM(input_size=embeds_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, hidden=None):
        input_embeds = self.embedding(input_ids)

        lstm_output, output_hidden = self.lstm(input_embeds, hidden)

        lstm_output = self.dropout(lstm_output)

        lm_output = self.lm_head(lstm_output)

        return lm_output, output_hidden


class TransformerModel(nn.Module):

    def __init__(self, vocab_size: int, hidden_dim: int, num_head: int, feedforward_dim: int,
                 num_layers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        encoder_layers = TransformerEncoderLayer(hidden_dim, num_head, feedforward_dim, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.hidden_dim = hidden_dim
        self.decoder = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        src = self.embedding(src) * math.sqrt(self.hidden_dim)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.dropout(output)
        output = self.decoder(output)
        return output


def generate_square_subsequent_mask(sz):
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


class PositionalEncoding(nn.Module):

    def __init__(self, hidden_dim: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2) * (-math.log(10000.0) / hidden_dim))
        pe = torch.zeros(max_len, 1, hidden_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.shape[0]]
        return self.dropout(x)


class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x


class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout):
        super(Attention, self).__init__()
        self.embed_dim = hidden_size
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim

        self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        max_positions = 10000
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.uint8)).view(
                1, 1, max_positions, max_positions
            ),
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        attn_weights = attn_weights / (value.size(-1) ** 0.5)

        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[:, :, key_length - query_length: key_length, :key_length].bool()
        attn_weights = torch.where(causal_mask, attn_weights, self.masked_bias.to(attn_weights.dtype))

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def forward(self, hidden_states, attention_mask):
        query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        attn_output, attn_weights = self._attn(query, key, value, attention_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output, attn_weights


class MLP(nn.Module):
    def __init__(self, intermediate_size, hidden_size, dropout):
        super().__init__()
        embed_dim = hidden_size
        self.c_fc = Conv1D(intermediate_size, embed_dim)
        self.c_proj = Conv1D(embed_dim, intermediate_size)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, inner_dim, num_heads, dropout):
        super(TransformerBlock, self).__init__()
        self.ln_1 = nn.LayerNorm(hidden_size, eps=1e-5)
        self.attn = Attention(hidden_size=hidden_size, num_heads=num_heads, dropout=dropout)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=1e-5)

        self.mlp = MLP(inner_dim, hidden_size, dropout)

    def forward(self, hidden_states, attention_mask):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(hidden_states, attention_mask)
        attn_output = attn_outputs[0]
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = residual + feed_forward_hidden_states

        return hidden_states


class Transformer(nn.Module):
    def __init__(self, vocab_size, hidden_size, inner_dim, num_heads, num_layers, dropout):
        super(Transformer, self).__init__()
        self.embed_dim = hidden_size
        self.embedding = nn.Embedding(vocab_size, self.embed_dim)
        self.pos_encoder = PositionalEncoding(hidden_size, dropout)
        self.drop = nn.Dropout(dropout)
        self.h = nn.ModuleList(
            [TransformerBlock(hidden_size, inner_dim, num_heads, dropout) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=1e-5)
        self.decoder = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        batch_size = input_ids.shape[0]
        if attention_mask is not None:
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=torch.long)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0
        inputs_embeds = self.embedding(input_ids)
        hidden_states = self.pos_encoder(inputs_embeds)

        for block in self.h:
            hidden_states = block(inputs_embeds, attention_mask)

        hidden_states = self.ln_f(hidden_states)
        output = self.decoder(hidden_states)

        return output
