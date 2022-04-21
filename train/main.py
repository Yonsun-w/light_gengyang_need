# -*- coding: utf-8 -*-
import numpy as np
import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as py_Dataset
import datetime
from config import read_config
from layers.ADSNet_model import ADSNet_Model
from layers.LightNet_model import LightNet_Model
from layers.DCNNet_model import DCNNet_Model
from layers.DCNLSTM_ADSNet_model import DCNLSTM_Model
from layers.DCN_ADSNet_model import DCNADSNet_Model
from layers.DCN_ADSNet_abl_model import DCNADSNetabl_Model
from layers.DCN_ADSNet_attn_model import DCNADSNet_attn_Model
from layers.DCN_ADSNet_lite_model import DCNADSNet_lite_Model
from layers.DCN_ADSNet_lite_attn_model import DCNADSNet_lite_attn_Model
from layers.DCN_ADSNet_lite_tf_model import DCNADSNet_lite_tf_Model
from layers.DCN_ADSNet_lite_tf2_model import DCNADSNet_lite_tf2_Model
from layers.DCN_ADSNet_lite_tf3_model import DCNADSNet_lite_tf3_Model
from layers.DCN_ADSNet_lite_tf4_model import DCNADSNet_lite_tf4_Model
from layers.DCN_ADSNet_lite_tf5_model import DCNADSNet_lite_tf5_Model
from layers.E3D_lstm import E3DLSTM_Model
from layers.ablation import Ablation_without_DCN, Ablation_without_transformer, Ablation_without_WBTE, Ablation_without_WandT
from generator import DataGenerator
from scores import Cal_params_epoch, Model_eval
from utils import Plot_res


def selectModel(config_dict):
    if config_dict['NetName'] == 'ADSNet':
        model = ADSNet_Model(obs_tra_frames=config_dict['TruthHistoryHourNum'], obs_channels=1,
                             wrf_tra_frames=config_dict['ForecastHourNum'],
                             wrf_channels=config_dict['WRFChannelNum'], config_dict=config_dict).to(
            config_dict['Device'])
    elif config_dict['NetName'] == 'LightNet':
        model = LightNet_Model(obs_tra_frames=config_dict['TruthHistoryHourNum'], obs_channels=1,
                               wrf_tra_frames=config_dict['ForecastHourNum'],
                               wrf_channels=config_dict['WRFChannelNum'], config_dict=config_dict).to(
            config_dict['Device'])
    elif config_dict['NetName'] == 'DCNNet':
        model = DCNNet_Model(obs_tra_frames=config_dict['TruthHistoryHourNum'], obs_channels=1,
                             wrf_tra_frames=config_dict['ForecastHourNum'],
                             wrf_channels=config_dict['WRFChannelNum'], config_dict=config_dict).to(
            config_dict['Device'])
    elif config_dict['NetName'] == 'DCNADSNet':
        model = DCNADSNet_Model(obs_tra_frames=config_dict['TruthHistoryHourNum'], obs_channels=1,
                                wrf_tra_frames=config_dict['ForecastHourNum'],
                                wrf_channels=config_dict['WRFChannelNum'], config_dict=config_dict).to(
            config_dict['Device'])
    elif config_dict['NetName'] == 'DCNADSNet_abl':
        model = DCNADSNetabl_Model(obs_tra_frames=config_dict['TruthHistoryHourNum'], obs_channels=1,
                                wrf_tra_frames=config_dict['ForecastHourNum'],
                                wrf_channels=config_dict['WRFChannelNum'], config_dict=config_dict).to(
            config_dict['Device'])
    elif config_dict['NetName'] == 'DCNLSTM_ALL':
        model = DCNLSTM_Model(obs_tra_frames=config_dict['TruthHistoryHourNum'], obs_channels=1,
                             wrf_tra_frames=config_dict['ForecastHourNum'],
                             wrf_channels=config_dict['WRFChannelNum'], config_dict=config_dict).to(
            config_dict['Device'])
    elif config_dict['NetName'] == 'DCNADSNet_attn':
        model = DCNADSNet_attn_Model(obs_tra_frames=config_dict['TruthHistoryHourNum'], obs_channels=1,
                                wrf_tra_frames=config_dict['ForecastHourNum'],
                                wrf_channels=config_dict['WRFChannelNum'], config_dict=config_dict).to(
            config_dict['Device'])
    elif config_dict['NetName'] == 'DCNADSNet_lite':
        model = DCNADSNet_lite_Model(obs_tra_frames=config_dict['TruthHistoryHourNum'], obs_channels=1,
                                wrf_tra_frames=config_dict['ForecastHourNum'],
                                wrf_channels=config_dict['WRFChannelNum'], config_dict=config_dict).to(
            config_dict['Device'])
    elif config_dict['NetName'] == 'DCNADSNet_lite_attn':
        model = DCNADSNet_lite_attn_Model(obs_tra_frames=config_dict['TruthHistoryHourNum'], obs_channels=1,
                                     wrf_tra_frames=config_dict['ForecastHourNum'],
                                     wrf_channels=config_dict['WRFChannelNum'], config_dict=config_dict).to(
            config_dict['Device'])
    elif config_dict['NetName'] == 'DCNADSNet_lite_tf':
        model = DCNADSNet_lite_tf_Model(obs_tra_frames=config_dict['TruthHistoryHourNum'], obs_channels=1,
                                     wrf_tra_frames=config_dict['ForecastHourNum'],
                                     wrf_channels=config_dict['WRFChannelNum'], config_dict=config_dict).to(
            config_dict['Device'])
    elif config_dict['NetName'] == 'DCNADSNet_lite_tf2':
        model = DCNADSNet_lite_tf2_Model(obs_tra_frames=config_dict['TruthHistoryHourNum'], obs_channels=1,
                                     wrf_tra_frames=config_dict['ForecastHourNum'],
                                     wrf_channels=config_dict['WRFChannelNum'], config_dict=config_dict).to(
            config_dict['Device'])
    elif config_dict['NetName'] == 'DCNADSNet_lite_tf3':
        model = DCNADSNet_lite_tf3_Model(obs_tra_frames=config_dict['TruthHistoryHourNum'], obs_channels=1,
                                     wrf_tra_frames=config_dict['ForecastHourNum'],
                                     wrf_channels=config_dict['WRFChannelNum'], config_dict=config_dict).to(
            config_dict['Device'])
    elif config_dict['NetName'] == 'DCNADSNet_lite_tf4':
        model = DCNADSNet_lite_tf4_Model(obs_tra_frames=config_dict['TruthHistoryHourNum'], obs_channels=1,
                                     wrf_tra_frames=config_dict['ForecastHourNum'],
                                     wrf_channels=config_dict['WRFChannelNum'], config_dict=config_dict).to(
            config_dict['Device'])
    elif config_dict['NetName'] == 'DCNADSNet_lite_tf5':
        model = DCNADSNet_lite_tf5_Model(obs_tra_frames=config_dict['TruthHistoryHourNum'], obs_channels=1,
                                     wrf_tra_frames=config_dict['ForecastHourNum'],
                                     wrf_channels=config_dict['WRFChannelNum'], config_dict=config_dict).to(
            config_dict['Device'])
    elif config_dict['NetName'] == 'Ablation_without_DCN':
        model = Ablation_without_DCN(obs_tra_frames=config_dict['TruthHistoryHourNum'], obs_channels=1,
                                     wrf_tra_frames=config_dict['ForecastHourNum'],
                                     wrf_channels=config_dict['WRFChannelNum'], config_dict=config_dict).to(
            config_dict['Device'])
    elif config_dict['NetName'] == 'Ablation_without_transformer':
        model = Ablation_without_transformer(obs_tra_frames=config_dict['TruthHistoryHourNum'], obs_channels=1,
                                     wrf_tra_frames=config_dict['ForecastHourNum'],
                                     wrf_channels=config_dict['WRFChannelNum'], config_dict=config_dict).to(
            config_dict['Device'])
    elif config_dict['NetName'] == 'Ablation_without_WBTE':
        model = Ablation_without_WBTE(obs_tra_frames=config_dict['TruthHistoryHourNum'], obs_channels=1,
                                     wrf_tra_frames=config_dict['ForecastHourNum'],
                                     wrf_channels=config_dict['WRFChannelNum'], config_dict=config_dict).to(
            config_dict['Device'])
    elif config_dict['NetName'] == 'Ablation_without_WandT':
        model = Ablation_without_WandT(obs_tra_frames=config_dict['TruthHistoryHourNum'], obs_channels=1,
                                     wrf_tra_frames=config_dict['ForecastHourNum'],
                                     wrf_channels=config_dict['WRFChannelNum'], config_dict=config_dict).to(
            config_dict['Device'])
    elif config_dict['NetName'] == 'E3D_LSTM':
        model = E3DLSTM_Model(obs_tra_frames=config_dict['TruthHistoryHourNum'], obs_channels=1,
                                     wrf_tra_frames=config_dict['ForecastHourNum'],
                                     wrf_channels=config_dict['WRFChannelNum'], config_dict=config_dict).to(
            config_dict['Device'])
    else:
        print('`{}` not support'.format(config_dict['NetName']))
        assert False
    return model


def DoTrain(config_dict):
    # data index
    TrainSetFilePath = 'data_index/TrainCase.txt'
    ValSetFilePath = 'data_index/ValCase.txt'
    TestSetFilePath = 'data_index/TestCase.txt'
    train_list = []
    with open(TrainSetFilePath) as file:
        for line in file:
            train_list.append(line.rstrip('\n').rstrip('\r\n'))
    val_list = []
    with open(ValSetFilePath) as file:
        for line in file:
            val_list.append(line.rstrip('\n').rstrip('\r\n'))
    test_list = []
    with open(TestSetFilePath) as file:
        for line in file:
            test_list.append(line.rstrip('\n').rstrip('\r\n'))
    # data
    train_data = DataGenerator(train_list, config_dict)
    train_loader = DataLoader(dataset=train_data, batch_size=config_dict['Batchsize'], shuffle=True, num_workers=0)
    val_data = DataGenerator(val_list, config_dict)
    val_loader = DataLoader(dataset=val_data, batch_size=config_dict['Batchsize'], shuffle=False, num_workers=0)
    test_data = DataGenerator(test_list, config_dict)
    test_loader = DataLoader(dataset=test_data, batch_size=config_dict['Batchsize'], shuffle=False, num_workers=0)

    # model
    model = selectModel(config_dict)

    # loss function
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(20))

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config_dict['LearningRate'])

    # eval
    model_eval_valdata = Model_eval(config_dict, is_save_model=True)
    model_eval_testdata = Model_eval(config_dict, is_save_model=False)

    # plot ETS
    ets_plot = Plot_res('./plot', ['val', 'test'], 'sumETS', 'epoch', 'sumETS', enable=True)

    print('Beginning train!')

    for epoch in range(config_dict['EpochNum']):
        # train_calparams_epoch = Cal_params_epoch()
        for i, (X, y) in enumerate(train_loader):
            wrf, obs = X
            label = y
            wrf = wrf.to(config_dict['Device'])
            obs = obs.to(config_dict['Device'])
            label = label.to(config_dict['Device'])

            pre_frames = model(wrf, obs)

            # backward
            optimizer.zero_grad()
            loss = criterion(torch.flatten(pre_frames), torch.flatten(label))
            loss.backward()

            # update weights
            optimizer.step()

            # output
            print('TRAIN INFO: epoch:{} ({}/{}) loss:{:.5f}'.format(epoch, i+1, len(train_loader), loss.item()))
            # pod, far, ts, ets = train_calparams_epoch.cal_batch(label, pre_frames)
            # sumpod, sumfar, sumts, sumets = train_calparams_epoch.cal_batch_sum(label, pre_frames)
            # info = 'TRAIN INFO: epoch:{} ({}/{}) loss:{:.5f}\nPOD:{:.5f}  FAR:{:.5f}  TS:{:.5f}  ETS:{:.5f}\nsumPOD:{:.5f}  sumFAR:{:.5f}  sumTS:{:.5f}  sumETS:{:.5f}\n'\
            #     .format(epoch, i+1, len(train_loader), loss.item(), pod, far, ts, ets, sumpod, sumfar, sumts, sumets)
            # print(info)

        val_sumets = model_eval_valdata.eval(val_loader, model, epoch)
        test_sumets = model_eval_testdata.eval(test_loader, model, epoch)
        ets_plot.step([val_sumets, test_sumets])

    # SelectEpoch(modelrecordname)


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'

    config_dict = read_config()

    if not os.path.isdir(config_dict['ModelFileDir']):
        os.makedirs(config_dict['ModelFileDir'])

    if not os.path.isdir(config_dict['RecordFileDir']):
        os.makedirs(config_dict['RecordFileDir'])

    if not os.path.isdir(config_dict['VisResultFileDir']):
        os.makedirs(config_dict['VisResultFileDir'])

    # train
    DoTrain(config_dict)



