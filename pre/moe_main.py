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
from scores2 import Cal_params_epoch
from layers.LightNetPlus.LightNet_plus import LightNet_plus_Model


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

def time_data_iscomplete(datetime_peroid, WRFFileDir, ForecastHourNum,TruthFileDirGrid, TruthHistoryHourNum):
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
                          row_col=config_dict['GridRowColNum'],wrf_channels=config_dict['WRFChannelNum'],
                          obs_channel=1).to(config_dict['Device'])
    elif config_dict['NetName'] =='LightNet+':
        model = LightNet_plus_Model(obs_tra_frames=config_dict['TruthHistoryHourNum'], obs_channels=1,
                                     wrf_tra_frames=config_dict['ForecastHourNum'],
                                     wrf_channels=config_dict['WRFChannelNum'], config_dict=config_dict).to(config_dict['Device'])

    else:
        print('`{}` not support'.format(config_dict['NetName']))
        assert False
    return model


def DoPredict(config_dict):
    print('pre,name={}'.format(config_dict['NetName']))

    TestSetFilePath = 'data_index/TestCase.txt'
    pre_list = []
    with open(TestSetFilePath) as file:
        for line in file:
            line = line.rstrip('\n').rstrip('\r\n')
            # 由于数据不全 所以需要校验数据的完整
            if time_data_iscomplete(line, WRFFileDir=config_dict['WRFFileDir'],
                                    ForecastHourNum=config_dict['ForecastHourNum'],
                                    TruthFileDirGrid=config_dict['TruthFileDirGrid'],
                                    TruthHistoryHourNum=config_dict['TruthHistoryHourNum']):

                pre_list.append(line.rstrip('\n').rstrip('\r\n'))

    print('加载数据完毕， val{}测试集,阈值={},file={}'.format(len(pre_list),config_dict['Threshold'],config_dict['ModelFilePath']))


    #########

    pre_data = DataGenerator(pre_list, config_dict)
    pre_loader = DataLoader(dataset=pre_data, batch_size=1, shuffle=True)
    model = selectModel(config_dict)
    model_file = torch.load(config_dict['ModelFilePath'], map_location=torch.device(config_dict['Device']))
    model.load_state_dict(model_file)
    model = model.to(config_dict['Device'])
    model.eval()
    vis = Torch_vis(config_dict, enable=False)


    POD = ''
    FAR = ''
    TS = ''
    ETS= ''

    nPOD = ''
    nFAR = ''
    nTS = ''
    nETS= ''

    print('-------------输出逐小时累加-------------------')

    # 输出逐小时 累加
    for t in range(config_dict['ForecastHourNum']):
        calparams_epoch = Cal_params_epoch(neighbor=1, threshold=config_dict['Threshold'])
        for i, (X, y) in enumerate(pre_loader):
            wrf, obs, obs_old = X
            label = y
            wrf = wrf.to(config_dict['Device'])
            obs = obs.to(config_dict['Device'])
            obs_old = obs_old.to(config_dict['Device'])
            label = label.to(config_dict['Device'])
            if config_dict['NetName'] == 'MOE':
                pre_frames = model(wrf, obs, obs_old)
            elif config_dict['NetName'] == 'LightNet+':
                pre_frames = model(wrf, obs)
            else:
                pre_frames, h = model(wrf, obs)
            pre_frames = pre_frames[:, 0:t + 1].contiguous()

            label = label[:, 0:t + 1].contiguous()

            eval_scores_neighbor = calparams_epoch.cal_batch_neighbor(label[:, :], pre_frames[:, :])
            eval_scores = calparams_epoch.cal_batch_sum(label[:, :], pre_frames[:, :])

            if torch.sum(label) > 0:
                vis.save_diff(pre_frames, label, i + 1)
                vis.save_oripre(pre_frames, i + 1)

        print('true frames={} forecast frames={},在{}小时到{}小时内的指标'.format(config_dict['TruthHistoryHourNum'],
                                                                        config_dict['ForecastHourNum'], t, t + 1))
        eval_scores = calparams_epoch.cal_epoch_sum()
        eval_scores_neighbor = calparams_epoch.cal_epoch_neighbor()
        info = 'TEST EPOCH INFO: \n' \
               'Point-> POD: {:.5f}  FAR: {:.5f}  TS: {:.5f}  ETS: {:.5f} FOM: {:.5f} BIAS: {:.5f} HSS: {:.5f} PC: {:.5f}\n' \
               'Neighbor-> POD: {:.5f}  FAR: {:.5f}  TS: {:.5f}  ETS: {:.5f} FOM: {:.5f} BIAS: {:.5f} HSS: {:.5f} PC: {:.5f}\n' \
            .format(eval_scores['POD'], eval_scores['FAR'], eval_scores['TS'], eval_scores['ETS'],
                    eval_scores['FOM'], eval_scores['BIAS'], eval_scores['HSS'], eval_scores['PC'],
                    eval_scores_neighbor['POD'], eval_scores_neighbor['FAR'], eval_scores_neighbor['TS'],
                    eval_scores_neighbor['ETS'],
                    eval_scores_neighbor['FOM'], eval_scores_neighbor['BIAS'], eval_scores_neighbor['HSS'],
                    eval_scores_neighbor['PC'])

        POD = POD + ' {:.5f}'.format(eval_scores['POD'])
        FAR = FAR + ' {:.5f}'.format(eval_scores['FAR'])
        TS = TS + ' {:.5f}'.format(eval_scores['TS'])
        ETS = ETS + ' {:.5f}'.format(eval_scores['ETS'])

        nPOD = nPOD + ' {:.5f}'.format(eval_scores_neighbor['POD'])
        nFAR = nFAR + ' {:.5f}'.format(eval_scores_neighbor['FAR'])
        nTS = nTS + ' {:.5f}'.format(eval_scores_neighbor['TS'])
        nETS = nETS + ' {:.5f}'.format(eval_scores_neighbor['ETS'])

    print(info)
    print(POD)
    print(FAR)
    print(TS)
    print(ETS)

    print('------------neighbor-----------------')

    print(nPOD)
    print(nFAR)
    print(nTS)
    print(nETS)


    POD = ''
    FAR = ''
    TS = ''
    ETS= ''

    nPOD = ''
    nFAR = ''
    nTS = ''
    nETS= ''

    print('-------------输出逐小时单独-------------------')


    # 输出逐小时
    for t in range(config_dict['ForecastHourNum']):
        calparams_epoch = Cal_params_epoch(neighbor=1, threshold=config_dict['Threshold'])

        for i, (X, y) in enumerate(pre_loader):
            wrf, obs,wrf_old = X
            label = y
            wrf_old = wrf_old.to(config_dict['Device'])
            wrf = wrf.to(config_dict['Device'])
            obs = obs.to(config_dict['Device'])
            label = label.to(config_dict['Device'])
            if config_dict['NetName'] == 'MOE' :
                pre_frames = model(wrf, obs, wrf_old)
            elif config_dict['NetName'] == 'LightNet+':
                pre_frames = model(wrf, obs)
            else :
                pre_frames, h = model(wrf, obs)
            pre_frames = pre_frames[:,t:t+1]
            label = label[:, t:t+1]
            eval_scores = calparams_epoch.cal_batch_sum(label[:, :], pre_frames[:, :])
            eval_scores_neighbor = calparams_epoch.cal_batch_neighbor(label[:, :], pre_frames[:, :])

            if torch.sum(label) > 0:
                vis.save_diff(pre_frames, label, i + 1)
                vis.save_oripre(pre_frames, i + 1)
            # # output
            if i % 100 == 0:
                info = 'TEST INFO: ({}/{}) \n{}\n{}\n'.format(i + 1, len(pre_loader), str(eval_scores),
                                                              str(eval_scores_neighbor))


            # print('TEST INFO: ({}/{})'.format(i + 1, len(pre_loader)))

        print('true frames={} forecast frames={},在{}小时到{}小时内的指标'.format(config_dict['TruthHistoryHourNum'],
                                                         config_dict['ForecastHourNum'], t, t+1))
        eval_scores = calparams_epoch.cal_epoch_sum()
        eval_scores_neighbor = calparams_epoch.cal_epoch_neighbor()
        info = 'TEST EPOCH INFO: \n' \
               'Point-> POD: {:.5f}  FAR: {:.5f}  TS: {:.5f}  ETS: {:.5f} FOM: {:.5f} BIAS: {:.5f} HSS: {:.5f} PC: {:.5f}\n' \
               'Neighbor-> POD: {:.5f}  FAR: {:.5f}  TS: {:.5f}  ETS: {:.5f} FOM: {:.5f} BIAS: {:.5f} HSS: {:.5f} PC: {:.5f}\n' \
            .format(eval_scores['POD'], eval_scores['FAR'], eval_scores['TS'], eval_scores['ETS'],
                    eval_scores['FOM'], eval_scores['BIAS'], eval_scores['HSS'], eval_scores['PC'],
                    eval_scores_neighbor['POD'], eval_scores_neighbor['FAR'], eval_scores_neighbor['TS'],
                    eval_scores_neighbor['ETS'],
                    eval_scores_neighbor['FOM'], eval_scores_neighbor['BIAS'], eval_scores_neighbor['HSS'],
                    eval_scores_neighbor['PC'])

        POD = POD + ' {:.5f}'.format(eval_scores['POD'])
        FAR = FAR + ' {:.5f}'.format(eval_scores['FAR'])
        TS = TS + ' {:.5f}'.format(eval_scores['TS'])
        ETS = ETS + ' {:.5f}'.format(eval_scores['ETS'])


        nPOD = nPOD + ' {:.5f}'.format(eval_scores_neighbor['POD'])
        nFAR = nFAR + ' {:.5f}'.format(eval_scores_neighbor['FAR'])
        nTS = nTS + ' {:.5f}'.format(eval_scores_neighbor['TS'])
        nETS = nETS + ' {:.5f}'.format(eval_scores_neighbor['ETS'])


    print('end----------1--------------------')
    print(info)
    print(POD)
    print(FAR)
    print(TS)
    print(ETS)

    print('------------neighbor-----------------')

    print(nPOD)
    print(nFAR)
    print(nTS)
    print(nETS)



if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    config_dict = read_config()

    if not os.path.isdir(config_dict['VisResultFileDir']):
        os.makedirs(config_dict['VisResultFileDir'])

    DoPredict(config_dict)



