import os
import torch
from torch.optim import AdamW
from tqdm import tqdm

from model.config import LSTMConfig, TrainingConfig
from model.BiLSTM_CRF import BiLSTM_CRF
from corpus import build
from BiLSTM.evaluating import Metrics

os.environ["CUDA_VISIBLE_DEVICES"] = "4"


def sort_by_lengths(input_ids, label):
    pairs = list(zip(input_ids, label))

    pairs.sort(key=lambda pair: len(pair[0]), reverse=True)

    input_ids, label = list(zip(*pairs))

    return input_ids, label


def get_tensor_data(batch_input_token, tokenizer):
    pad = '<pad>'
    max_len = len(batch_input_token[0])

    valid_length = [len(x) for x in batch_input_token]
    valid_length = torch.tensor(valid_length)
    batch_input_token = [x + [pad] * (max_len - len(x)) for x in batch_input_token]

    batch_input_ids = tokenizer.batch_encode(batch_input_token)
    batch_input_ids = torch.tensor(batch_input_ids)

    return batch_input_ids, valid_length


def indexed(targets, tagset_size, start_id):
    """将targets中的数转化为在[T*T]大小序列中的索引,T是标注的种类"""
    batch_size, max_len = targets.size()
    for col in range(max_len - 1, 0, -1):
        targets[:, col] += (targets[:, col - 1] * tagset_size)
    targets[:, 0] += (start_id * tagset_size)
    return targets


def lstm_crf_loss(crf_score, target, tokenizer, real_length):
    pad_ids = tokenizer.encode('<pad>')
    bos_ids = tokenizer.encode('<bos>')
    eos_ids = tokenizer.encode('<eos>')

    device = crf_score.device

    batch_size, seq_len = target.shape
    t_size = len(tokenizer)

    mask = (target != pad_ids)
    real_length = mask.sum(dim=1)
    target = indexed(target, t_size, bos_ids)

    target = target.masked_select(mask)
    flatten_score = crf_score.masked_select(
        mask.view(batch_size, seq_len, 1, 1).expand_as(crf_score)
    ).view(-1, t_size * t_size).contiguous()
    golden_score = flatten_score.gather(
        dim=1, index=target.unsqueeze(1)).sum()

    # scores_upto_t[i, j]表示第i个句子的第t个词被标注为j标记的所有t时刻事前的所有子路径的分数之和
    scores_upto_t = torch.zeros(batch_size, t_size).to(device)
    for t in range(seq_len):
        # 当前时刻 有效的batch_size（因为有些序列比较短)
        batch_size_t = (real_length > t).sum().item()
        if t == 0:
            scores_upto_t[:batch_size_t] = crf_score[:batch_size_t, t, bos_ids, :]
        else:
            # We add scores at current timestep to scores accumulated up to previous
            # timestep, and log-sum-exp Remember, the cur_tag of the previous
            # timestep is the prev_tag of this timestep
            # So, broadcast prev. timestep's cur_tag scores
            # along cur. timestep's cur_tag dimension
            scores_upto_t[:batch_size_t] = torch.logsumexp(
                crf_score[:batch_size_t, t, :, :] +
                scores_upto_t[:batch_size_t].unsqueeze(2),
                dim=1
            )
    all_path_scores = scores_upto_t[:, eos_ids].sum()

    # 训练大约两个epoch loss变成负数，从数学的角度上来说，loss = -logP
    loss = (all_path_scores - golden_score) / sum(real_length)
    return loss


def train(model, train_input_token, train_label, valid_input_token, valid_label, tokenizer, label2id):
    train_input_token, train_label = sort_by_lengths(train_input_token, train_label)
    valid_input_token, valid_label = sort_by_lengths(valid_input_token, valid_label)
    total_step = len(train_input_token) // batch_size + 1

    best_val_loss = float('inf')
    val_cnt = 0
    for epoch in range(epochs):
        step = 0
        total_loss = 0.

        pbar = tqdm(range(0, len(train_input_token), batch_size), desc="Epoch {}".format(epoch + 1))
        for idx in pbar:
            batch_input_token = train_input_token[idx:idx + batch_size]
            batch_label = train_label[idx:idx + batch_size]

            model.train()
            batch_input_ids, real_length = get_tensor_data(batch_input_token, tokenizer)
            batch_label_ids, _ = get_tensor_data(batch_label, label2id)

            batch_input_ids = batch_input_ids.to(device)
            batch_label_ids = batch_label_ids.to(device)

            scores = model(batch_input_ids, real_length)

            loss = loss_func(scores, batch_label_ids, label2id, real_length)
            total_loss += loss.item()
            step += 1

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            pbar.set_postfix({'average loss': total_loss / step,
                              'lr': optimizer.state_dict()['param_groups'][0]['lr']})

            if step % (total_step // 2) == 0 and step != 0:
                model.eval()
                with torch.no_grad():
                    valid_step = 0
                    valid_total_loss = 0.
                    for v_idx in range(0, len(valid_input_token), batch_size):
                        valid_batch_input_token = valid_input_token[v_idx:v_idx + batch_size]
                        valid_batch_label = valid_label[v_idx:v_idx + batch_size]

                        valid_batch_input_ids, valid_real_length = get_tensor_data(valid_batch_input_token, tokenizer)
                        valid_batch_label_ids, _ = get_tensor_data(valid_batch_label, label2id)

                        valid_batch_input_ids = valid_batch_input_ids.to(device)
                        valid_batch_label_ids = valid_batch_label_ids.to(device)

                        valid_scores = model(valid_batch_input_ids, valid_real_length)
                        loss = loss_func(valid_scores, valid_batch_label_ids, label2id, real_length)

                        valid_total_loss += loss.item()
                        valid_step += 1
                average_loss = valid_total_loss / valid_step
                if average_loss < best_val_loss:
                    best_val_loss = average_loss
                    print("\nModel at step {}, loss: {}".format(step, average_loss))
                else:
                    val_cnt += 1
                    if val_cnt == 5:
                        print("Valid loss doesn't decline for 3 valid steps, training stops!")
                        break
        if val_cnt == 5:
            break


filename = './ChineseCorpus199801.txt'
train_data, train_label, valid_data, valid_label, test_data, test_label, tokenizer, label_dict = build(filename)

vocab_size = len(tokenizer)
num_label = len(label_dict)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embeds_size = LSTMConfig.embeds_size
hidden_size = LSTMConfig.hidden_size
n_layers = LSTMConfig.n_layers
crf = LSTMConfig.crf

model = BiLSTM_CRF(vocab_size=vocab_size, embeds_size=embeds_size, hidden_size=hidden_size,
                   n_layers=n_layers, num_label=num_label)
model = model.to(device)
loss_func = lstm_crf_loss

epochs = TrainingConfig.epochs
lr = TrainingConfig.lr
batch_size = TrainingConfig.batch_size
optimizer = AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.999)

test_data, test_label = sort_by_lengths(test_data, test_label)
pbar = tqdm(range(0, len(test_data), batch_size), desc="Testing...")
pred = []
gold = []
for idx in pbar:
    test_batch_input_token = test_data[idx:idx + batch_size]
    test_batch_label = test_label[idx:idx + batch_size]

    test_batch_input_ids, test_real_length = get_tensor_data(test_batch_input_token, tokenizer)

    test_batch_input_ids = test_batch_input_ids.to(device)

    model.eval()
    with torch.no_grad():
        test_batch_decode_result = model.decode(test_batch_input_ids, test_real_length, label_dict)

    for i, result_idx in enumerate(test_batch_decode_result):
        label_list = []
        for j in range(test_real_length[i] - 1):
            label_list.append(label_dict.decode(result_idx[j].item()))
        pred.append(label_list)
    gold.extend(test_batch_label)
gold = [x[:-1] for x in gold]

metrics = Metrics(gold, pred)
metrics.report_scores()
metrics.report_confusion_matrix()

train(model, train_data, train_label, valid_data, valid_label, tokenizer, label_dict)

test_data, test_label = sort_by_lengths(test_data, test_label)
pbar = tqdm(range(0, len(test_data), batch_size), desc="Testing...")
pred = []
gold = []
for idx in pbar:
    test_batch_input_token = test_data[idx:idx + batch_size]
    test_batch_label = test_label[idx:idx + batch_size]

    test_batch_input_ids, test_real_length = get_tensor_data(test_batch_input_token, tokenizer)

    test_batch_input_ids = test_batch_input_ids.to(device)

    model.eval()
    with torch.no_grad():
        test_batch_decode_result = model.decode(test_batch_input_ids, test_real_length, label_dict)

    for i, result_idx in enumerate(test_batch_decode_result):
        label_list = []
        for j in range(test_real_length[i] - 1):
            label_list.append(label_dict.decode(result_idx[j].item()))
        pred.append(label_list)
    gold.extend(test_batch_label)
gold = [x[:-1] for x in gold]

metrics = Metrics(gold, pred)
metrics.report_scores()
metrics.report_confusion_matrix()
