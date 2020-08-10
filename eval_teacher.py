from json import load
from dataset import NewsDataset, TestDataset
from model import Student, Teacher
from dataset import NewsDataset, TestTeacherNewsDataset
import torch
from torch.utils import data

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class TeacherEvaluator:
    def __init__(self, hparams, path):
        self.hparams = hparams
        self.ds = TestTeacherNewsDataset(hparams['aspect_init_file'], path,
                          hparams['student']['pretrained'], hparams['maxlen'])
        self.idx2asp = torch.tensor(self.ds.get_idx2asp()).to(device)
        self.asp_cnt = self.params['student']['num_aspect']
        self.teacher = Teacher(self.idx2asp, self.asp_cnt, self.hparams['general_asp']).to(device)

    def test(self):
        loader = data.DataLoader(self.ds, batch_size=64, num_workers=10)
        z = torch.ones((self.asp_cnt, len(self.ds.asp2id))).to(device)
        texts, dates = [], []
        idx, score = [], []
        for batch in loader:
            bow, text, date = batch
            bow = bow.to(device)
            logits = self.teacher(bow, z)
            val, ans = torch.sort(logits, dim=-1, descending=True)
            idx.extend(ans.detach().tolist())
            score.extend(val.detach().tolist())
            texts.extend(text)
            dates.extend(date)
        return texts, idx, score, dates


if __name__ == '__main__':
    from config import hparams
    evaluaator = TeacherEvaluator(hparams, ['../data/tw/tw_fake_news_20190524_2020702_seg_full.json'])
    texts, dates, idx, score = evaluaator.test()
    n = '\n'
    list(map(lambda x: print(f'{x[0].split(n)[0]}\n{x[1]}\n{x[2]}\n{x[3]}\n') , zip(texts, dates, idx, score)))