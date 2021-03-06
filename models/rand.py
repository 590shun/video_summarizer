import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from summarizer.models import Trainer

"""
Randomにスコアの予測を行い,ベースライン手法との比較を行う実装.
"""

class Random(nn.Module):
    def __init__(self):
        super(Random, self).__init__()

    def forward(self, x):
        """
         [入力]
        x: (seq_len, batch_size, input_size)
        [出力]
        probs: (seq_len, batch_size, 1)
        """
        seq_len, batch_size, _ = x.shape
        scores = torch.rand((seq_len, batch_size, 1))
        scores = scores.to(x.device)
        return scores

class RandomTrainer(Trainer):
    def _init_model(self):
        model = Random()
        return model

    def train(self, fold):
        self.model.train()
        train_keys, _ = self._get_train_test_keys(fold)
        self.draw_gtscores(fold, train_keys)

        criterion = nn.MSELoss()
        if self.hps.use_cuda:
            criterion = criterion.cuda()
        
        # 最も良かったepochの評価指標を記録する
        best_corr, best_avg_f_score, best_max_f_score = -1.0, 0.0, 0.0

        # 各epochの処理
        for epoch in range(self.hps.epochs):
            train_avg_loss = []
            dist_scores = {}
            random.shuffle(train_keys)

            # 各動画に対するtraining
            for key in train_keys:
                dataset = self.dataset[key]
                seq = dataset["features"][...]
                seq = torch.from_numpy(seq).unsqueeze(1)
                target = dataset["gtscore"][...]
                target = torch.from_numpy(target).view(-1, 1, 1)

                # frame scoreの正規化
                target -= target.min()
                target /= target.max() - target.min()

                if self.hps.use_cuda:
                    seq, target = seq.cuda(), target.cuda()

                scores = self.model(seq)
                loss = criterion(scores, target)
                train_avg_loss.append(float(loss))
                dist_scores[key] = scores.detach().cpu().numpy()

            # 各epochのtraining lossの平均を取る
            train_avg_loss = np.mean(np.array(train_avg_loss))
            self.log.info(f"Epoch: {f'{epoch+1}/{self.hps.epochs}':6}   "
                            f"Loss: {train_avg_loss:.05f}")
            self.hps.writer.add_scalar(f"{self.dataset_name}/Fold_{fold+1}/Train/Loss", train_avg_loss, epoch)

            # Evaluate performances on test keys
            if epoch % self.hps.test_every_epochs == 0:
                avg_corr, (avg_f_score, max_f_score) = self.test(fold)
                self.model.train()
                self.hps.writer.add_scalar(f"{self.dataset_name}/Fold_{fold+1}/Test/Correlation", avg_corr, epoch)
                self.hps.writer.add_scalar(f"{self.dataset_name}/Fold_{fold+1}/Test/F-score_avg", avg_f_score, epoch)
                self.hps.writer.add_scalar(f"{self.dataset_name}/Fold_{fold+1}/Test/F-score_max", max_f_score, epoch)
                best_avg_f_score = max(best_avg_f_score, avg_f_score)
                best_max_f_score = max(best_max_f_score, max_f_score)
                if avg_corr > best_corr:
                    best_corr = avg_corr
                    self.best_weights = self.model.state_dict()

        # Log final scores
        self.draw_scores(fold, dist_scores)
        
        return best_corr, best_avg_f_score, best_max_f_score


if __name__ == "__main__":
    model = Random()
    print("Trainable parameters in model:", sum([_.numel() for _ in model.parameters() if _.requires_grad]))

    x = torch.randn(10, 3, 1024)
    y = model(x)
    assert x.shape[0] == y.shape[0]
    assert x.shape[1] == y.shape[1]
    assert y.shape[2] == 1
