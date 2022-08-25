import torch
import datetime
import os

class Cal_params_epoch(object):
    def __init__(self, neighbor, threshold=0.5):
        # n1->TP  n2->FP  n3->FN  n4->TN
        self.n1sum = 0
        self.n2sum = 0
        self.n3sum = 0
        self.n4sum = 0
        self.n1sum_neighbor = 0
        self.n2sum_neighbor = 0
        self.n3sum_neighbor = 0
        self.n4sum_neighbor = 0
        self.neighbor = neighbor
        self.eps = 1e-10
        self.threshold = threshold

        # 带阈值的
    # def _transform_sum(self, y_true, y_pred):
    #     # y_true = y_true.permute(1, 0, 2, 3, 4).cpu().contiguous()
    #     y_true = y_true.permute(1, 0, 2, 3).cpu().contiguous()
    #     y_pred = y_pred.permute(1, 0, 2, 3, 4).cpu().contiguous()
    #     # y_pred = torch.sigmoid(y_pred)
    #     if 0 <= self.threshold <= 1:
    #         y_pred[y_pred > self.threshold] = 1
    #         y_pred[y_pred < 1] = 0
    #     else:
    #         y_pred = torch.round(y_pred)
    #     frames = y_true.shape[0]
    #     sum_true = torch.zeros(y_true[0].shape)
    #     sum_pred = torch.zeros(y_pred[0].shape)
    #     for i in range(frames):
    #         sum_true += y_true[i]
    #         sum_pred += y_pred[i]
    #     sum_true = torch.flatten(sum_true)
    #     sum_pred = torch.flatten(sum_pred)
    #     return sum_true, sum_pred

    def _transform_sum(self, y_true, y_pred):
        # y_true = y_true.permute(1, 0, 2, 3, 4).cpu().contiguous()
        y_true = y_true.permute(1, 0, 2, 3).cpu().contiguous()
        y_pred = y_pred.permute(1, 0, 2, 3, 4).cpu().contiguous()
        y_pred = torch.round(torch.sigmoid(y_pred))
        frames = y_true.shape[0]
        sum_true = torch.zeros(y_true[0].shape)
        sum_pred = torch.zeros(y_pred[0].shape)
        for i in range(frames):
            sum_true += y_true[i]
            sum_pred += y_pred[i]
        sum_true = torch.flatten(sum_true)
        sum_pred = torch.flatten(sum_pred)
        return sum_true, sum_pred




    def _transform_sum_neighbor(self, y_true, y_pred):
        y_true = y_true.permute(1, 0, 2, 3).cpu().contiguous()
        y_pred = y_pred.permute(1, 0, 2, 3, 4).cpu().contiguous()
        y_true = torch.reshape(y_true, y_pred.shape)
        y_true_neighbor = torch.zeros_like(y_true)
        frames = y_true.shape[1]
        if 0 <= self.threshold <= 1:
            y_pred[y_pred > self.threshold] = 1
            y_pred[y_pred < 1] = 0
        else:
            y_pred = torch.round(y_pred)
        mn = y_true.shape[2]
        for i in range(mn):
            for j in range(mn):
                il = i - self.neighbor if i - self.neighbor > 0 else 0
                ir = i + self.neighbor + 1 if i + self.neighbor + 1 < mn else mn
                jl = j - self.neighbor if j - self.neighbor > 0 else 0
                jr = j + self.neighbor + 1 if j + self.neighbor + 1 < mn else mn
                # print(y_true[:, :, i, j].shape, y_true[:, :, il:ir, jl:jr].shape, torch.sum(y_true[:, :, il:ir, jl:jr], dim=[2, 3]).shape)
                y_true_neighbor[:, :, i, j] = torch.sum(y_true[:, :, il:ir, jl:jr], dim=[2, 3])
                # print(y_true[:, :, i, j].shape, torch.sum(y_true[:, :, il:ir, jl, jr], dim=[0, 1]).shape)
        sum_true = torch.zeros(y_true_neighbor[0].shape)
        sum_pred = torch.zeros(y_pred[0].shape)
        for i in range(frames):
            sum_true += y_true_neighbor[i]
            sum_pred += y_pred[i]
        sum_true = torch.flatten(sum_true)
        sum_pred = torch.flatten(sum_pred)
        return sum_true, sum_pred

    def _POD_(self, n1, n3):

        return torch.div(n1, n1 + n3 + self.eps)

    def _FAR_(self, n1, n2):
        return torch.div(n2, n1 + n2 + self.eps)

    def _TS_(self, n1, n2, n3):
        return torch.div(n1, n1 + n2 + n3 + self.eps)

    def _ETS_(self, n1, n2, n3, n4):
        r = torch.div((n1 + n2) * (n1 + n3), n1 + n2 + n3 + n4 + self.eps)
        return torch.div(n1 - r, n1 + n2 + n3 - r + self.eps)

    def _FOM_(self, n1, n3):
        return torch.div(n3, n1 + n3 + self.eps)

    def _BIAS_(self, n1, n2, n3):
        return torch.div(n1 + n2, n1 + n3 + self.eps)

    def _HSS_(self, n1, n2, n3, n4):
        return torch.div(2 * (n1 * n4 - n2 * n3), (n1 + n3) * (n3 + n4) + (n1 + n2) * (n2 + n4) + self.eps)

    def _PC_(self, n1, n2, n3, n4):
        return torch.div(n1 + n4, n1 + n2 + n3 + n4 + self.eps)

    def _all_eval(self, n1, n2, n3, n4):
        pod = self._POD_(n1, n3)
        far = self._FAR_(n1, n2)
        ts = self._TS_(n1, n2, n3)
        ets = self._ETS_(n1, n2, n3, n4)
        fom = self._FOM_(n1, n3)
        bias = self._BIAS_(n1, n2, n3)
        hss = self._HSS_(n1, n2, n3, n4)
        pc = self._PC_(n1, n2, n3, n4)
        return {'POD': pod, 'FAR': far, 'TS': ts, 'ETS': ets, 'FOM': fom, 'BIAS': bias, 'HSS': hss, 'PC': pc}

    def cal_batch_sum(self, y_true, y_pred):
        y_true, y_pred = self._transform_sum(y_true, y_pred)
        n1 = torch.sum((y_pred > 0) & (y_true > 0))
        n2 = torch.sum((y_pred > 0) & (y_true < 1))
        n3 = torch.sum((y_pred < 1) & (y_true > 0))
        n4 = torch.sum((y_pred < 1) & (y_true < 1))
        self.n1sum += n1
        self.n2sum += n2
        self.n3sum += n3
        self.n4sum += n4
        return self._all_eval(n1, n2, n3, n4)

    def cal_batch_neighbor(self, y_true, y_pred):
        y_true, y_pred = self._transform_sum_neighbor(y_true, y_pred)
        n1 = torch.sum((y_pred > 0) & (y_true > 0))
        n2 = torch.sum((y_pred > 0) & (y_true < 1))
        n3 = torch.sum((y_pred < 1) & (y_true > 0))
        n4 = torch.sum((y_pred < 1) & (y_true < 1))
        self.n1sum_neighbor += n1
        self.n2sum_neighbor += n2
        self.n3sum_neighbor += n3
        self.n4sum_neighbor += n4
        return self._all_eval(n1, n2, n3, n4)

    def cal_epoch_sum(self):

        print('strict', self.n1sum, self.n2sum, self.n3sum, self.n4sum)
        return self._all_eval(self.n1sum, self.n2sum, self.n3sum, self.n4sum)

    def cal_epoch_neighbor(self):
        print('neighbor',self.n1sum_neighbor, self.n2sum_neighbor, self.n3sum_neighbor, self.n4sum_neighbor)
        return self._all_eval(self.n1sum_neighbor, self.n2sum_neighbor, self.n3sum_neighbor, self.n4sum_neighbor)

if __name__ == "__main__":
    pass