from config import hparams
import torch
import json
from torch.utils import data
from model import Student, Teacher
from dataset import TestNewsDataset, NewsDataset
import logging
import os
from sklearn.metrics import f1_score
from tqdm import tqdm


torch.set_printoptions(profile="full")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# torch.cuda.set_device(0)
logging.basicConfig(level=logging.INFO)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
logging.info(f'current_gpu: {torch.cuda.get_device_name(torch.cuda.current_device())}, {torch.cuda.current_device()}')


class Evaluator:
    def __init__(self, model_path, epoch, test_file) -> None:
        self.hparams = self.load_config(model_path)
        logging.info('loading dataset...')
        self.ds = TestNewsDataset(self.hparams['aspect_init_file'], test_file, maxlen=hparams['maxlen'], pretrained=hparams['student']['pretrained'])
        logging.info('finished')
        self.loader = data.DataLoader(self.ds, batch_size=50, num_workers=5)
        logging.info('loadding model...')
        self.load_model(model_path, epoch)
        logging.info('finished')
        self.student.eval()
    
    def load_config(self, dir):
        with open(os.path.join(dir, 'config.json')) as f:
            return json.load(f)


    def load_model(self, model, epoch):
        model = os.path.join(model, f'epoch_{epoch}_student.pt')
        self.student = Student(hparams['student']).to(device)
        self.student.load_state_dict(torch.load(model)['student'])
        # return student
    
    def test(self):
        texts = []
        category = []
        score = []
        dates = []
        for batch in tqdm(self.loader):
            idx, text, date  = batch
            idx = idx.to(device)
            with torch.no_grad():
                logits = self.student(idx)
            val, ans = torch.sort(logits, dim=-1, descending=True)
            val = val.detach().cpu().tolist()
            ans = ans.detach().cpu().tolist()
            category.extend(ans)
            score.extend(val)
            texts.extend(text)
            dates.extend(date)

        return texts, category, score, dates
    

if __name__ == '__main__':
    evaluator = Evaluator('ckpt/news_lite_fake_title/', 1, ['../data/tw/tw_fake_news_20190524_2020702_seg_full.json'])
    result = evaluator.test()
    new = '\n'
    for r in list(map(lambda t, c, s: print(f'{t.split(new)[0]}\n{c}\n{s}\n'), result[0], result[1], result[2])):
        pass
