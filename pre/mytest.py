import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import datetime
from config import read_config
from moe_main_generator import DataGenerator
from layers.moe.moe_ADSNet import ADSNet_Model
from layers.moe.moe_LightNet import LightNet_Model
from layers.moe.MOE import MOE_Model
# from layers.E3D_lstm import E3DLSTM_Model
from vision import showimg, mutishow, Torch_vis
# from scores import Cal_params_epoch
from test_socre import Cal_params_epoch,Model_eval


def getTimePeriod(dt):
    time = dt.strftime("%H:%M:%S")
    hour = int(time[0:2])
    if 0 <= hour < 6:
        nchour = '00'
    elif 6 <= hour < 12:
        nchour = '06'
    elif 12 <= hour < 18:
        nchour = '12'
    elif 18 <= hour <= 23:
        nchour = '18'
    else:
        print('error')
    delta_hour = hour - int(nchour)
    return nchour, delta_hour


def time_data_iscomplete(datetime_peroid, WRFFileDir, ForecastHourNum, TruthFileDirGrid, TruthHistoryHourNum):
    datetime_peroid = datetime_peroid.rstrip('\n')
    datetime_peroid = datetime_peroid.rstrip('\r\n')
    if datetime_peroid == '':
        return False
    is_complete = True

    ddt = datetime.datetime.strptime(datetime_peroid, '%Y%m%d%H%M')
    # read WRF
    utc = ddt + datetime.timedelta(hours=-8)
    ft = utc + datetime.timedelta(hours=(-6))
    nchour, delta_hour = getTimePeriod(ft)
    delta_hour += 6

    filepath = os.path.join(WRFFileDir, ft.date().strftime("%Y%m%d"), str(nchour))

    if not os.path.exists(filepath):
        is_complete = False

    # read labels
    for hour_plus in range(ForecastHourNum):
        dt = ddt + datetime.timedelta(hours=hour_plus)
        tFilePath = TruthFileDirGrid + dt.strftime('%Y%m%d%H%M') + '.npy'
        if not os.path.exists(tFilePath):
            is_complete = False

    # read history observations
    for hour_plus in range(TruthHistoryHourNum):
        dt = ddt + datetime.timedelta(hours=hour_plus - TruthHistoryHourNum)
        tFilePath = TruthFileDirGrid + dt.strftime('%Y%m%d%H%M') + '.npy'
        if not os.path.exists(tFilePath):
            is_complete = False

    # 这一部分的作用是，获取我们 历史那段时间 也就是和预测时间相同长度时间wrf
    ddt = datetime.datetime.strptime(datetime_peroid, '%Y%m%d%H%M') + datetime.timedelta(
        hours=config_dict['TruthHistoryHourNum'])
    utc = ddt + datetime.timedelta(hours=-8)
    ft = utc + datetime.timedelta(hours=(-6))
    nchour, delta_hour = getTimePeriod(ft)
    delta_hour += 6
    filepath = os.path.join(config_dict['WRFFileDir'], ft.date().strftime("%Y%m%d"), str(nchour))
    if not os.path.exists(filepath):
        is_complete = False

    return is_complete


def selectModel(config_dict):
    if config_dict['NetName'] == 'ADSNet':
        model = ADSNet_Model(obs_tra_frames=config_dict['TruthHistoryHourNum'], obs_channels=1,
                             wrf_tra_frames=config_dict['ForecastHourNum'],
                             wrf_channels=config_dict['WRFChannelNum'], row_col=config_dict['GridRowColNum']).to(
            config_dict['Device'])
    elif config_dict['NetName'] == 'LightNet':
        model = LightNet_Model(obs_tra_frames=config_dict['TruthHistoryHourNum'], obs_channels=1,
                               wrf_tra_frames=config_dict['ForecastHourNum'],
                               wrf_channels=config_dict['WRFChannelNum'], row_col=config_dict['GridRowColNum']).to(
            config_dict['Device'])
    elif config_dict['NetName'] == 'MOE':
        model = MOE_Model(truth_history_hour_num=config_dict['TruthHistoryHourNum'],
                          forecast_hour_num=config_dict['ForecastHourNum'],
                          row_col=config_dict['GridRowColNum'], wrf_channels=config_dict['WRFChannelNum'],
                          obs_channel=1, ads_net_model_path='models/moe/ADSNet_model_maxETS.pkl',
                          light_net_model_path='models/moe/LightNet_model_maxETS.pkl').to(config_dict['Device'])

    else:
        print('`{}` not support'.format(config_dict['NetName']))
        assert False
    return model


def DoPredict(config_dict):
    print('pre,name={}'.format(config_dict['NetName']))
    # TestSetFilePath = 'data_index/ValCase.txt'
    TestSetFilePath = 'data_index/TestCase.txt'
    pre_list = []
    with open(TestSetFilePath) as file:
        for line in file:
            pre_list.append(line.rstrip('\n').rstrip('\r\n'))

    #########

    pre_list = []

    # st = datetime.datetime.strptime('2019071000', '%Y%m%d%H')
    # et = datetime.datetime.strptime('2019101000', '%Y%m%d%H')
    #
    # print('加载从{}到{}之间的数据集'.format(st, et))
    # while st <= et:
    #     line = datetime.datetime.strftime(st, '%Y%m%d%H%M')
    #     # 由于数据不全 所以需要校验数据的完整
    #     if time_data_iscomplete(line,  WRFFileDir=config_dict['WRFFileDir'],ForecastHourNum=config_dict['ForecastHourNum'],
    #                             TruthFileDirGrid=config_dict['TruthFileDirGrid'], TruthHistoryHourNum=config_dict['TruthHistoryHourNum']):
    #         pre_list.append(line.rstrip('\n').rstrip('\r\n'))
    #
    #     st += datetime.timedelta(hours=3)

    TestSetFilePath = 'data_index/TestCase.txt'
    pre_list = []
    with open(TestSetFilePath) as file:
        for line in file:
            pre_list.append(line.rstrip('\n').rstrip('\r\n'))

    print('加载数据完毕， val{}测试集'.format(len(pre_list)))

    #########

    pre_data = DataGenerator(pre_list, config_dict)
    pre_loader = DataLoader(dataset=pre_data, batch_size=1, shuffle=True)
    model = selectModel(config_dict)
    model_file = torch.load(config_dict['ModelFilePath'], map_location=torch.device(config_dict['Device']))
    model.load_state_dict(model_file)
    model = model.to(config_dict['Device'])
    model.eval()


    model_eval_valdata = Model_eval(config_dict)

    for t in range(config_dict['ForecastHourNum']):
        print(t)
        model_eval_valdata.eval(pre_loader, model,t)


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'

    config_dict = read_config()

    if not os.path.isdir(config_dict['VisResultFileDir']):
        os.makedirs(config_dict['VisResultFileDir'])

    DoPredict(config_dict)



