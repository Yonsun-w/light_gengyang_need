import torch
import datetime
import os

class Cal_params_epoch(object):
    def __init__(self, threshold):
        # n1->TP  n2->FP  n3->FN  n4->TN
        self.n1 = 0
        self.n2 = 0
        self.n3 = 0
        self.n4 = 0
        self.n1sum = 0
        self.n2sum = 0
        self.n3sum = 0
        self.n4sum = 0
        self.eps = 1e-10
        self.threshold = threshold

    def _transform_sum(self, y_true, y_pred):
        # y_true = y_true.permute(1, 0, 2, 3, 4).cpu().contiguous()
        y_true = y_true.permute(1, 0, 2, 3).cpu().contiguous()
        y_pred = y_pred.permute(1, 0, 2, 3, 4).cpu().contiguous()
        # y_pred = torch.sigmoid(y_pred)
        if 0 <= self.threshold <= 1:
            y_pred[y_pred > self.threshold] = 1
            y_pred[y_pred < 1] = 0
        else:
            y_pred = torch.round(y_pred)
        frames = y_true.shape[0]
        sum_true = torch.zeros(y_true[0].shape)
        sum_pred = torch.zeros(y_pred[0].shape)
        for i in range(frames):
            sum_true += y_true[i]
            sum_pred += y_pred[i]
        sum_true = torch.flatten(sum_true)
        sum_pred = torch.flatten(sum_pred)
        return sum_true, sum_pred

    def _transform(self, y_true, y_pred):
        y_true = y_true.cpu()
        y_pred = y_pred.cpu()
        y_true = torch.flatten(y_true)
        y_pred = torch.flatten(y_pred)
        if 0 <= self.threshold <= 1:
            y_pred[y_pred > self.threshold] = 1
            y_pred[y_pred < 1] = 0
        else:
            y_pred = torch.round(y_pred)
        return y_true, y_pred

    def _POD_(self, n1, n3):
        return torch.div(n1, n1 + n3 + self.eps)

    def _FAR_(self, n1, n2):
        return torch.div(n2, n1 + n2 + self.eps)

    def _TS_(self, n1, n2, n3):
        return torch.div(n1, n1 + n2 + n3 + self.eps)

    def _ETS_(self, n1, n2, n3, r):
        return torch.div(n1 - r, n1 + n2 + n3 - r + self.eps)

    def cal_batch(self, y_true, y_pred):
        y_true, y_pred = self._transform(y_true, y_pred)
        n1 = torch.sum((y_pred > 0) & (y_true > 0))
        n2 = torch.sum((y_pred > 0) & (y_true < 1))
        n3 = torch.sum((y_pred < 1) & (y_true > 0))
        n4 = torch.sum((y_pred < 1) & (y_true < 1))
        r = torch.div((n1 + n2) * (n1 + n3), n1 + n2 + n3 + n4)
        pod = self._POD_(n1, n3)
        far = self._FAR_(n1, n2)
        ts = self._TS_(n1, n2, n3)
        ets = self._ETS_(n1, n2, n3, r)
        self.n1 += n1
        self.n2 += n2
        self.n3 += n3
        self.n4 += n4
        return pod, far, ts, ets

    def cal_batch_sum(self, y_true, y_pred):
        y_true, y_pred = self._transform_sum(y_true, y_pred)
        n1 = torch.sum((y_pred > 0) & (y_true > 0))
        n2 = torch.sum((y_pred > 0) & (y_true < 1))
        n3 = torch.sum((y_pred < 1) & (y_true > 0))
        n4 = torch.sum((y_pred < 1) & (y_true < 1))
        r = torch.div((n1 + n2) * (n1 + n3), n1 + n2 + n3 + n4)
        pod = self._POD_(n1, n3)
        far = self._FAR_(n1, n2)
        ts = self._TS_(n1, n2, n3)
        ets = self._ETS_(n1, n2, n3, r)
        self.n1sum += n1
        self.n2sum += n2
        self.n3sum += n3
        self.n4sum += n4
        return pod, far, ts, ets

    def cal_epoch(self):
        r = torch.div((self.n1 + self.n2) * (self.n1 + self.n3), self.n1 + self.n2 + self.n3 + self.n4)
        pod = self._POD_(self.n1, self.n3)
        far = self._FAR_(self.n1, self.n2)
        ts = self._TS_(self.n1, self.n2, self.n3)
        ets = self._ETS_(self.n1, self.n2, self.n3, r)
        return pod, far, ts, ets

    def cal_epoch_sum(self):
        r = torch.div((self.n1sum + self.n2sum) * (self.n1sum + self.n3sum), self.n1sum + self.n2sum + self.n3sum + self.n4sum)
        pod = self._POD_(self.n1sum, self.n3sum)
        far = self._FAR_(self.n1sum, self.n2sum)
        ts = self._TS_(self.n1sum, self.n2sum, self.n3sum)
        ets = self._ETS_(self.n1sum, self.n2sum, self.n3sum, r)
        return pod, far, ts, ets

if __name__ == "__main__":
    pass