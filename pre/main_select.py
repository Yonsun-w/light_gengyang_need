import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import datetime
from config import read_config
from generator import DataGenerator
from layers.ADSNet_model import ADSNet_Model
from layers.LightNet_model import LightNet_Model
from layers.DCNNet_model import DCNNet_Model
from layers.DCN_ADSNet_model import DCNADSNet_Model
from layers.DCN_ADSNet_attn_model import DCNADSNet_attn_Model
from layers.DCN_ADSNet_lite_model import DCNADSNet_lite_Model
from layers.DCN_ADSNet_lite_tf_model import DCNADSNet_lite_tf_Model
from layers.DCN_ADSNet_lite_tf2_model import DCNADSNet_lite_tf2_Model
from layers.DCN_ADSNet_lite_tf3_model import DCNADSNet_lite_tf3_Model
from vision import showimg, mutishow, Torch_vis
from scores import Cal_params_epoch


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

    else:
        print('`{}` not support'.format(config_dict['NetName']))
        assert False
    return model

def DoPredict(config_dict):
    TestSetFilePath = 'data_index/TestCase.txt'
    pre_list = []
    with open(TestSetFilePath) as file:
        for line in file:
            pre_list.append(line.rstrip('\n').rstrip('\r\n'))
    pre_data = DataGenerator(pre_list, config_dict)
    pre_loader = DataLoader(dataset=pre_data, batch_size=1, shuffle=False)
    model = selectModel(config_dict)
    model_file = torch.load(config_dict['ModelFilePath'], map_location=torch.device(config_dict['Device']))
    model.load_state_dict(model_file)
    model = model.to(config_dict['Device'])

    vis = Torch_vis(config_dict, enable=False)

    threshold = [0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8]
    calparams_epoch = []
    for ts in threshold:
        calparams_epoch.append(Cal_params_epoch(ts))



    # for i, (X, y) in enumerate(pre_loader):
    #     wrf, obs = X
    #     label = y
    #     wrf = wrf.to(config_dict['Device'])
    #     obs = obs.to(config_dict['Device'])
    #     label = label.to(config_dict['Device'])
    #
    #     pre_frames = model(wrf, obs)
    #     pre_frames = torch.sigmoid(pre_frames)
    #
    #     if torch.sum(label) > 1000:
    #         vis.save_diff(pre_frames, label, i)
    #     # output
    #     for j in range(len(threshold)):
    #         # pod, far, ts, ets = calparams_epoch[j].cal_batch(label, pre_frames)
    #         sumpod, sumfar, sumts, sumets = calparams_epoch[j].cal_batch_sum(label[:, :], pre_frames[:, :])
    #     # info = 'TEST INFO: ({}/{}) \nPOD:{:.5f}  FAR:{:.5f}  TS:{:.5f}  ETS:{:.5f}\nsumPOD:{:.5f}  sumFAR:{:.5f}  sumTS:{:.5f}  sumETS:{:.5f}\n' \
    #     #     .format(i + 1, len(pre_loader), pod, far, ts, ets, sumpod, sumfar, sumts, sumets)
    #     print('TEST INFO: ({}/{})'.format(i + 1, len(pre_loader)))
    # for j, ts in enumerate(threshold):
    #     sumpod, sumfar, sumts, sumets = calparams_epoch[j].cal_epoch_sum()
    #     info = 'TEST EPOCH INFO (threshold={}): \nsumPOD:{:.5f}  sumFAR:{:.5f}  sumTS:{:.5f}  sumETS:{:.5f}\n'.format(ts, sumpod, sumfar, sumts, sumets)
    #     print(info)



if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    config_dict = read_config()

    if not os.path.isdir(config_dict['VisResultFileDir']):
        os.makedirs(config_dict['VisResultFileDir'])

    DoPredict(config_dict)


