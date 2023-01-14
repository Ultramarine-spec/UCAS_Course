import torch
import torch.nn as nn
from itertools import zip_longest
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embeds_size, hidden_size, n_layers, num_label):
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embeds_size)
        self.bilstm = nn.LSTM(embeds_size, hidden_size, batch_first=True, bidirectional=True, num_layers=n_layers)
        self.linear = nn.Linear(hidden_size * 2, num_label)

    def forward(self, input_ids, real_length):
        input_embeds = self.embedding(input_ids)

        packed = pack_padded_sequence(input_embeds, real_length, batch_first=True)

        lstm_output, _ = self.bilstm(packed)

        lstm_output, _ = pad_packed_sequence(lstm_output, batch_first=True)

        output = self.linear(lstm_output)

        return output


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, embeds_size, hidden_size, n_layers, num_label):
        super(BiLSTM_CRF, self).__init__()
        self.bilstm = BiLSTM(vocab_size, embeds_size, hidden_size, n_layers, num_label)

        self.crf_transition = nn.Parameter(torch.ones(num_label, num_label) * 1 / num_label)

    def forward(self, input_ids, real_length):
        output = self.bilstm(input_ids, real_length)

        batch_size, max_len, num_label = output.shape

        # 计算CRF scores, 这个scores大小为[B, L, out_size, out_size]
        # 也就是每个字对应对应一个 [out_size, out_size]的矩阵
        # 这个矩阵第i行第j列的元素的含义是：上一时刻tag为i，这一时刻tag为j的分数
        crf_score = output.unsqueeze(2).expand(-1, -1, num_label, -1) + self.crf_transition

        return crf_score

    def decode(self, input_ids, real_length, label_dict):
        bos_idx = label_dict.encode('<bos>')
        eos_idx = label_dict.encode('<eos>')
        pad_idx = label_dict.encode('<pad>')
        label_num = len(label_dict)

        crf_score = self.forward(input_ids, real_length)
        device = crf_score.device
        batch_size, max_len, num_state = crf_score.shape[:3]

        viterbi = torch.zeros(batch_size, max_len, num_state).to(device)
        real_length = torch.LongTensor(real_length).to(device)

        path = torch.zeros(batch_size, max_len, num_state).long().to(device)

        for step in range(max_len):
            batch_flag = (real_length > step).sum().item()
            if step == 0:
                viterbi[:batch_flag, step, :] = crf_score[:batch_flag, step, bos_idx, :]
                path[:batch_flag, step, :] = bos_idx
            else:
                # 第n个batch，在step时刻到状态i的最可能路径概率的log为：viterbi[n, step, i]
                # 第n个batch，在step时刻，从状态i到状态j的转移概率与状态j对应的观测概率的乘积的log为：crf_score[n, step, i, j]
                max_score, prev_state = torch.max(viterbi[:batch_flag, step - 1, :].unsqueeze(2) + \
                                                  crf_score[:batch_flag, step, :, :], dim=1)
                viterbi[:batch_flag, step, :] = max_score
                path[:batch_flag, step, :] = prev_state

        path = path.view(batch_size, -1)
        result = []
        state_t = None
        for step in range(max_len - 1, 0, -1):
            batch_flag = (real_length > step).sum().item()
            if step == max_len - 1:
                idx = torch.ones(batch_flag).long() * step * label_num
                idx = idx.to(device)
                idx += eos_idx
            else:
                prev_batch_flag = len(state_t)

                new_in_batch = torch.LongTensor([eos_idx] * (batch_flag - prev_batch_flag)).to(device)

                offset = torch.cat([state_t, new_in_batch], dim=0)

                idx = torch.ones(batch_flag).long() * step * len(label_dict)
                idx = idx.to(device)
                idx += offset.long()

            state_t = path[:batch_flag].gather(dim=1, index=idx.unsqueeze(1).long())
            state_t = state_t.squeeze(1)
            result.append(state_t.tolist())

        result = list(zip_longest(*reversed(result), fillvalue=pad_idx))
        result = torch.tensor(result).long()

        return result
