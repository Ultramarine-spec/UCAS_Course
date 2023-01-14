from sklearn.metrics import classification_report


class HMM:
    def __init__(self):
        self.chars = set()

        self.init_p = {'B': 0.5, 'M': 0, 'E': 0, 'S': 0.5}
        self.state_cnt = {'B': 0, 'M': 0, 'E': 0, 'S': 0}
        self.state_ch_cnt = {}
        self.state_transfer_cnt = {}

        self.state_transfer_matrix = dict()
        self.observation_matrix = dict()

    def init_matrix(self):
        for st1 in ['B', 'M', 'E', 'S']:
            self.state_transfer_matrix.setdefault(st1, {})
            for st2 in ['B', 'M', 'E', 'S']:
                self.state_transfer_matrix[st1].setdefault(st2, 0.)

        for st in ['B', 'M', 'E', 'S']:
            self.observation_matrix.setdefault(st, {})
            for ch in self.chars:
                self.observation_matrix[st].setdefault(ch, 1.0 / float(self.state_cnt[st]))

    def train(self, data_path):
        print('Training...')

        def get_state(i, length):
            if length == 1:
                return 'S'
            if i == 0:
                return 'B'
            if i == length - 1:
                return 'E'
            return 'M'

        with open(data_path, 'r', encoding='gbk') as f:
            data = [x.strip() for x in f.readlines()]
            data = [x for x in data if len(x) != 0]
            data = data[len(data) // 10:]
            f.close()
        states_list = []
        start_with_B = 0
        start_with_S = 0
        for line in data:
            if len(line) == 0:
                continue
            sen_seg = line.split('  ')[1:]
            sen_seg = [x.split('/')[0] for x in sen_seg]
            states = ''
            for word in sen_seg:
                for i in range(len(word)):
                    ch = word[i]
                    self.chars.add(ch)
                    state = get_state(i, len(word))
                    states += state
                    self.state_cnt[state] += 1
                    self.state_ch_cnt.setdefault(state, {})
                    self.state_ch_cnt[state].setdefault(ch, 0)
                    self.state_ch_cnt[state][ch] += 1
            if states[0] == 'B':
                start_with_B += 1
            elif states[0] == 'S':
                start_with_S += 1
            else:
                raise Exception('Sentence should start with S or B')

            states_list.append(states)

        self.init_p['B'] = float(start_with_B) / float(start_with_B + start_with_S)
        self.init_p['S'] = float(start_with_S) / float(start_with_B + start_with_S)

        for sts in states_list:
            for i in range(len(sts) - 1):
                self.state_transfer_cnt.setdefault(sts[i], {})
                self.state_transfer_cnt[sts[i]].setdefault(sts[i + 1], 0)
                self.state_transfer_cnt[sts[i]][sts[i + 1]] += 1

        self.init_matrix()

        for st1, st1_next_state_dic in self.state_transfer_cnt.items():
            sum_cnt = sum(st1_next_state_dic.values())
            for st2 in st1_next_state_dic.keys():
                self.state_transfer_matrix[st1][st2] = float(st1_next_state_dic[st2]) / float(sum_cnt)

        for st, st_ch_dic in self.state_ch_cnt.items():
            for ch in st_ch_dic.keys():
                self.observation_matrix[st][ch] = float(st_ch_dic[ch]) / float(self.state_cnt[st])

    def test(self, sens):
        pred = []
        pred_out = []

        for sen in sens:
            sen = sen.strip()
            if len(sen) == 0:
                continue
            viterbi = []
            path = []
            for t in range(len(sen)):
                o = sen[t]
                viterbi.append({})
                path.append({})
                for st in ['B', 'M', 'E', 'S']:

                    if o not in self.observation_matrix[st].keys():
                        self.observation_matrix[st][o] = 1.0 / float(self.state_cnt[st])

                    if t == 0:
                        viterbi[t][st] = self.init_p[st] * self.observation_matrix[st][o]
                        path[t][st] = [st]
                    else:
                        viterbi[t][st] = 0
                        path[t][st] = []
                        for pre_st in ['B', 'M', 'E', 'S']:
                            prob = viterbi[t - 1][pre_st] * self.state_transfer_matrix[pre_st][st] * \
                                   self.observation_matrix[st][o]
                            if prob >= viterbi[t][st]:
                                viterbi[t][st] = prob
                                path[t][st] = path[t - 1][pre_st] + [st]
            prob = 0
            state = ''
            for st in ['B', 'M', 'E', 'S']:
                if viterbi[len(sen) - 1][st] >= prob:
                    prob = viterbi[len(sen) - 1][st]
                    state = st
            result = path[len(sen) - 1][state]
            pred.append(result)
            # print(' '.join(result))

            output = ""
            for i in range(len(result)):
                if result[i] == 'S' or result[i] == 'E':
                    output += sen[i] + ' '
                else:
                    output += sen[i]
            pred_out.append(output)

            # print(output)
        return pred, pred_out


def get_label(sen_seg):
    label = []
    for token in sen_seg:
        if len(token) == 1:
            label.append('S')
        elif len(token) == 2:
            label.extend('BE')
        else:
            label.extend('B' + 'M' * (len(token) - 2) + 'E')
    return label


if __name__ == '__main__':
    hmm = HMM()
    hmm.train('../ChineseCorpus199801.txt')
    with open('../ChineseCorpus199801.txt', 'r', encoding='gbk') as f:
        test_data = [x.strip() for x in f.readlines()]
        test_data = [x for x in test_data if len(x) != 0]
        test_data = test_data[:len(test_data) // 10]
        f.close()
    sen_seg = [x.split('  ')[1:] for x in test_data]
    sen_list = [''.join([x.split('/')[0] for x in sen]) for sen in sen_seg]
    sen_seg = [[x.split('/')[0] for x in sen] for sen in sen_seg]
    gold = [get_label(x) for x in sen_seg]

    pred, output = hmm.test(sen_list)
    # metrics = Metrics(gold, pred)
    # metrics.print_score()
    print('HMM')
    print(classification_report(sum(gold, []), sum(pred, []), digits=4))
