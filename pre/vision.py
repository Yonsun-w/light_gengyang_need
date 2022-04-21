import os
import numpy as np
import cv2
import torch
import imageio
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize

def show(img):
    # img = img.cpu().detach().numpy()
    plt.imshow(img)
    plt.show()
    plt.close()

def show2(img1, img2):
    plt.figure(1)
    plt.sca(plt.subplot(1, 2, 1))
    plt.imshow(img1)
    plt.sca(plt.subplot(1, 2, 2))
    plt.imshow(img2)
    plt.show()
    plt.close()



def showimg(img, light, p):
    img = img.cpu().detach().numpy()
    plt.imshow(img)
    for i in range(light.shape[0]):
        for j in range(light.shape[1]):
            if light[i, j] > 0.5:
                plt.plot(j, i, 'r*')
    # plt.show()
    savefile = p + '.jpg'
    plt.savefig(savefile)
    plt.close()

def mutishow(pre, light, his_light, p):
    pre = pre.cpu().detach().numpy()
    light = light.cpu().detach().numpy()
    his_light = his_light.cpu().detach().numpy()
    plt.figure(1)
    for i in range(his_light.shape[0]):
        tp = plt.subplot(1, 2, i + 1)
        plt.sca(tp)
        plt.imshow(his_light[i])
    for i in range(pre.shape[0]):
        tp = plt.subplot(1, 2, i + 1 + 1)
        plt.sca(tp)
        for y in range(light[i].shape[0]):
            for x in range(light[i].shape[1]):
                if light[i, y, x] > 0.5 and pre[i, y, x] < 0.5:
                    pre[i, y, x] = 2
                if light[i, y, x] > 0.5 and pre[i, y, x] > 0.5:
                    pre[i, y, x] = 3
        plt.imshow(pre[i])
    plt.show()
    plt.close()


class Plot_res(object):
    def __init__(self, plot_save_dir, plot_datainfo, plot_title, plot_xname, plot_yname, enable=False):
        self.data = []
        self.plot_save_dir = plot_save_dir
        self.plot_datainfo = plot_datainfo
        self.plot_title = plot_title
        self.plot_xname = plot_xname
        self.plot_yname = plot_yname
        self.enable = enable

    def step(self, data):
        assert len(data) == len(self.plot_datainfo)
        self.data.append(data)
        if len(self.data) % 5 == 0:
            self.save()

    def save(self):
        if self.enable:
            data = np.array(self.data)
            for i in range(len(self.plot_datainfo)):
                x = range(len(self.data))
                plt.plot(x, data[:, i], label=self.plot_datainfo[i])
            plt.title(self.plot_title)
            plt.xlabel(self.plot_xname)
            plt.ylabel(self.plot_yname)
            if not os.path.exists(self.plot_save_dir):
                os.makedirs(self.plot_save_dir)
            plt.legend()
            plt.savefig(os.path.join(self.plot_save_dir, '{}.jpg'.format(''.join(self.plot_datainfo))))
            plt.close()




class Torch_vis(object):
    def __init__(self, config_dict, enable=False):
        self.enable = enable
        self.vis_save_dir = config_dict['VisResultFileDir']
        self.label_save_dir = os.path.join(self.vis_save_dir, 'label')
        self.wrf_save_dir = os.path.join(self.vis_save_dir, 'wrf')
        self.pre_save_dir = os.path.join(self.vis_save_dir, 'pre')
        self.oripre_save_dir = os.path.join(self.vis_save_dir, 'oripre')
        self.diff_save_dir = os.path.join(self.vis_save_dir, 'diff')
        self.xy = config_dict['GridRowColNum']

    def _norm(self, img):
        mx = np.max(img)
        mn = np.min(img)
        if mx == mn:
            return img
        else:
            img = (img - mn) / (mx - mn)
        return img * 255

    def _filter_4dimtensor(self, tensor):
        kernel_size = (3, 3)
        for i in range(tensor.shape[0]):
            for j in range(tensor.shape[1]):
                tensor[i, j] = cv2.blur(tensor[i, j], kernel_size)
        return tensor

    def _label_trans(self, label):
        label = label.cpu().detach()
        label = label.squeeze().reshape([label.shape[0], label.shape[1], self.xy, self.xy])
        label = label.numpy()
        return label

    def _pre_trans(self, pre):
        pre = pre.cpu().detach()
        pre = pre.squeeze().reshape([pre.shape[0], pre.shape[1], self.xy, self.xy])
        pre = pre.numpy()
        return np.round(pre)

    def _oripre_trans(self, pre):
        pre = pre.cpu().detach()
        pre = pre.squeeze().reshape([pre.shape[0], pre.shape[1], self.xy, self.xy])
        pre = pre.numpy()
        pre = self._filter_4dimtensor(pre)
        return pre

    def save_label(self, label, info, epoch=None):
        if self.enable:
            if epoch != None:
                savedir = os.path.join(self.diff_save_dir, 'epoch={}'.format(epoch))
            else:
                savedir = self.label_save_dir
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            label = self._label_trans(label)
            for batch_num in range(label.shape[0]):
                for frame_num in range(label.shape[1]):
                    # img = self._norm(label[batch_num, frame_num])
                    img = np.zeros([label.shape[2], label.shape[3], 3], dtype=np.uint8)
                    for i in range(label.shape[2]):
                        for j in range(label.shape[3]):
                            if label[batch_num, frame_num, i, j] > 0.1:
                                img[i, j, 2] = 255
                            else:
                                img[i, j, 0] = 255
                                img[i, j, 1] = 255
                                img[i, j, 2] = 255
                    cv2.imwrite(os.path.join(savedir, '{}:{}_{}.jpg'.format(info, batch_num, frame_num + 1)), img)

    def save_wrf(self, wrf, info, epoch=None):
        if self.enable:
            for cl in range(wrf.shape[-1]):
                wrf_img = wrf[:, :, :, :, cl]
                if epoch != None:
                    savedir = os.path.join(self.diff_save_dir, 'epoch={}'.format(epoch), 'channel={}'.format(cl))
                else:
                    savedir = os.path.join(self.diff_save_dir, 'channel={}'.format(cl))
                if not os.path.exists(savedir):
                    os.makedirs(savedir)
                wrf_img = self._oripre_trans(wrf_img)
                for batch_num in range(wrf_img.shape[0]):
                    for frame_num in range(wrf_img.shape[1]):
                        img = self._norm(wrf_img[batch_num, frame_num])
                        cv2.imwrite(os.path.join(savedir, '{}:{}_{}.jpg'.format(info, batch_num, frame_num + 1)), img)

    def save_pre(self, pre, info, epoch=None):
        if self.enable:
            if epoch != None:
                savedir = os.path.join(self.diff_save_dir, 'epoch={}'.format(epoch))
            else:
                savedir = self.pre_save_dir
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            pre = self._pre_trans(pre)
            for batch_num in range(pre.shape[0]):
                for frame_num in range(pre.shape[1]):
                    img = self._norm(pre[batch_num, frame_num])
                    img_red = np.zeros([img.shape[0], img.shape[1], 3], dtype=np.uint8)
                    img_red[:, :, 0] = img
                    img_red[:, :, 1] = img
                    img_red = 255 - img_red
                    cv2.imwrite(os.path.join(savedir, '{}:{}_{}.jpg'.format(info, batch_num, frame_num + 1)), img_red)

    def save_oripre(self, pre, info, epoch=None):
        if self.enable:
            if epoch != None:
                savedir = os.path.join(self.diff_save_dir, 'epoch={}'.format(epoch))
            else:
                savedir = self.oripre_save_dir
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            pre = self._oripre_trans(pre)
            for batch_num in range(pre.shape[0]):
                for frame_num in range(pre.shape[1]):
                    fig = plt.figure()
                    # gs = gridspec.GridSpec(1, 1)
                    # gs.update(wspace=0.01, hspace=0.01, top=0.95, bottom=0.05, left=0.17, right=0.845)
                    norm = Normalize(vmin=0.0, vmax=1.0)
                    y_pred = pre[batch_num, frame_num]
                    # ax = plt.subplot(gs[0, 0])
                    ax = plt.subplot()
                    ax.imshow(y_pred, alpha=0.9, norm=norm, cmap=plt.cm.get_cmap('jet'))
                    # ax.imshow(y_pred, norm=norm, cmap=plt.cm.get_cmap('Wistia'))
                    cs = ax.contour(y_pred, norm=norm, levels=[0.5], linewidths=1, colors='k', linestyles='dashed')
                    ax.clabel(cs, inline=1, fmt='%.1f', fontsize=10)
                    # ax.imshow(y_pred, cmap=plt.cm.get_cmap('hsv'))
                    ax.set_xlim([0, pre.shape[-1]-1])
                    ax.set_ylim([0, pre.shape[-1]-1])
                    ax.xaxis.set_visible(False)
                    ax.yaxis.set_visible(False)
                    ax.margins(x=0, y=0)
                    # plt.savefig(os.path.join(savedir, '{}:{}_{}.jpg'.format(info, batch_num, frame_num + 1)), bbox_inches='tight', pad_inches=0.02)
                    # plt.show()
                    plt.savefig(os.path.join(savedir, '{}:{}_{}.jpg'.format(info, batch_num, frame_num + 1)), bbox_inches='tight', pad_inches=0.0)
                    plt.close()

    def save_diff(self, pre, label, info, epoch=None):
        if self.enable:
            if epoch != None:
                savedir = os.path.join(self.diff_save_dir, 'epoch={}'.format(epoch))
            else:
                savedir = self.diff_save_dir
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            pre = self._pre_trans(pre).astype(np.uint8)
            label = self._label_trans(label).astype(np.uint8)
            for batch_num in range(label.shape[0]):
                for frame_num in range(label.shape[1]):
                    diff = np.zeros([label.shape[2], label.shape[3], 3], dtype=np.uint8)
                    for i in range(label.shape[2]):
                        for j in range(label.shape[3]):
                            if label[batch_num, frame_num, i, j] == 1 and pre[batch_num, frame_num, i, j] == 0:
                                # blue -> N3
                                diff[i, j, 0] = 255
                            elif label[batch_num, frame_num, i, j] == 0 and pre[batch_num, frame_num, i, j] == 1:
                                # green -> N2
                                diff[i, j, 1] = 255
                            elif label[batch_num, frame_num, i, j] == 1 and pre[batch_num, frame_num, i, j] == 1:
                                # red -> N1
                                diff[i, j, 2] = 255
                            elif label[batch_num, frame_num, i, j] == 0 and pre[batch_num, frame_num, i, j] == 0:
                                # white -> N4
                                diff[i, j, 0] = 255
                                diff[i, j, 1] = 255
                                diff[i, j, 2] = 255
                            else:
                                print('-----maybe error-----{}: {}|{}'.
                                      format(info, label[batch_num, frame_num, i, j], pre[batch_num, frame_num, i, j]))
                    cv2.imwrite(os.path.join(savedir, '{}:{}_{}.jpg'.format(info, batch_num, frame_num + 1)), diff)





if __name__ == "__main__":
    mutishow(0,0,0,0)
