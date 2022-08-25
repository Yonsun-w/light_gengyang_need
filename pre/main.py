import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import datetime
from config import read_config
from generator import DataGenerator
from layers.ADSNet_model import ADSNet_Model
from layers.LightNet_model import LightNet_Model
#from layers.DCNNet_model import DCNNet_Model
#from layers.DCN_ADSNet_model import DCNADSNet_Model
#from layers.DCN_ADSNet_attn_model import DCNADSNet_attn_Model
#from layers.DCN_ADSNet_lite_model import DCNADSNet_lite_Model
#from layers.DCN_ADSNet_lite_tf_model import DCNADSNet_lite_tf_Model
#from layers.DCN_ADSNet_lite_tf2_model import DCNADSNet_lite_tf2_Model
#from layers.DCN_ADSNet_lite_tf3_model import DCNADSNet_lite_tf3_Model
from layers.OnlyObsNet_model import OnlyObsNet_Model
from layers.OnlyWRFNet_model import OnlyWRFNet_Model
#from layers.ablation import Ablation_without_DCN, Ablation_without_transformer, Ablation_without_WBTE, Ablation_without_WandT
# from layers.E3D_lstm import E3DLSTM_Model
from vision import showimg, mutishow, Torch_vis
# from scores import Cal_params_epoch
from layers.moe.MOE import MOE_Model
from scores2 import Cal_params_epoch

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
    elif config_dict['NetName'] == 'MOE':
        model = MOE_Model(truth_history_hour_num=config_dict['TruthHistoryHourNum'],
                          forecast_hour_num=config_dict['ForecastHourNum'],
                          row_col=config_dict['GridRowColNum'],wrf_channels=config_dict['WRFChannelNum'],
                          obs_channel=1, ads_net_model_path = 'models/moe/ADSNet_model_maxETS.pkl',
                 light_net_model_path='models/moe/LightNet_model_maxETS.pkl').to(config_dict['Device'])


    else:
        print('`{}` not support'.format(config_dict['NetName']))
        assert False
    return model


def DoPredict(config_dict):
    # TestSetFilePath = 'data_index/ValCase.txt'
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
    model.eval()

    vis = Torch_vis(config_dict, enable=False)

    calparams_epoch = Cal_params_epoch(neighbor=1, threshold=config_dict['Threshold'])

    POD = ''
    FAR = ''
    TS = ''
    ETS= ''

    nPOD = ''
    nFAR = ''
    nTS = ''
    nETS= ''
    # # 输出逐小时
    # for t in range(config_dict['ForecastHourNum']):
    #     for i, (X, y) in enumerate(pre_loader):
    #         wrf, obs = X
    #         label = y
    #         wrf = wrf.to(config_dict['Device'])
    #         obs = obs.to(config_dict['Device'])
    #         label = label.to(config_dict['Device'])
    #         pre_frames = model(wrf, obs)
    #         pre_frames = torch.sigmoid(pre_frames)
    #         pre_frames = pre_frames[:,t:t+1]
    #         label = label[:, t:t+1]
    #         eval_scores = calparams_epoch.cal_batch_sum(label[:, :], pre_frames[:, :])
    #         eval_scores_neighbor = calparams_epoch.cal_batch_neighbor(label[:, :], pre_frames[:, :])
    #
    #         # eval_scores = calparams_epoch.cal_batch_sum(label[:, :6], pre_frames[:, :6])
    #         # eval_scores_neighbor = calparams_epoch.cal_batch_neighbor(label[:, :6], pre_frames[:, :6])
    #         # eval_scores = calparams_epoch.cal_batch_sum(label[:, 6:], pre_frames[:, 6:])
    #         # eval_scores_neighbor = calparams_epoch.cal_batch_neighbor(label[:, 6:], pre_frames[:, 6:])
    #
    #         if torch.sum(label) > 0:
    #             vis.save_diff(pre_frames, label, i + 1)
    #             vis.save_oripre(pre_frames, i + 1)
    #             # vis.save_pre(pre_frames, i + 1)
    #             # vis.save_wrf(wrf, i)
    #         # output
    #         if i % 100 == 0:
    #             info = 'TEST INFO: ({}/{}) \n{}\n{}\n'.format(i + 1, len(pre_loader), str(eval_scores),
    #                                                           str(eval_scores_neighbor))
    #             print(info)
    #
    #         # print('TEST INFO: ({}/{})'.format(i + 1, len(pre_loader)))
    #
    #     print('true frames={} forecast frames={},在{}小时到{}小时内的指标'.format(config_dict['TruthHistoryHourNum'],
    #                                                      config_dict['ForecastHourNum'], t, t+1))
    #     eval_scores = calparams_epoch.cal_epoch_sum()
    #     eval_scores_neighbor = calparams_epoch.cal_epoch_neighbor()
    #     info = 'TEST EPOCH INFO: \n' \
    #            'Point-> POD: {:.5f}  FAR: {:.5f}  TS: {:.5f}  ETS: {:.5f} FOM: {:.5f} BIAS: {:.5f} HSS: {:.5f} PC: {:.5f}\n' \
    #            'Neighbor-> POD: {:.5f}  FAR: {:.5f}  TS: {:.5f}  ETS: {:.5f} FOM: {:.5f} BIAS: {:.5f} HSS: {:.5f} PC: {:.5f}\n' \
    #         .format(eval_scores['POD'], eval_scores['FAR'], eval_scores['TS'], eval_scores['ETS'],
    #                 eval_scores['FOM'], eval_scores['BIAS'], eval_scores['HSS'], eval_scores['PC'],
    #                 eval_scores_neighbor['POD'], eval_scores_neighbor['FAR'], eval_scores_neighbor['TS'],
    #                 eval_scores_neighbor['ETS'],
    #                 eval_scores_neighbor['FOM'], eval_scores_neighbor['BIAS'], eval_scores_neighbor['HSS'],
    #                 eval_scores_neighbor['PC'])
    #
    #     POD = POD + ' {:.5f}'.format(eval_scores['POD'])
    #     FAR = FAR + ' {:.5f}'.format(eval_scores['FAR'])
    #     TS = TS + ' {:.5f}'.format(eval_scores['TS'])
    #     ETS = ETS + ' {:.5f}'.format(eval_scores['ETS'])
    #     print(info)
    #     print(POD)
    #     print(FAR)
    #     print(TS)
    #     print(ETS)
    #
    #     print('neighbor')
    #
    #     nPOD = nPOD + ' {:.5f}'.format(eval_scores_neighbor['POD'])
    #     nFAR = nFAR + ' {:.5f}'.format(eval_scores_neighbor['FAR'])
    #     nTS = nTS + ' {:.5f}'.format(eval_scores_neighbor['TS'])
    #     nETS = nETS + ' {:.5f}'.format(eval_scores_neighbor['ETS'])
    #
    #     print(nPOD)
    #     print(nFAR)
    #     print(nTS)
    #     print(nETS)



    # 输出累加
    for t in range(config_dict['ForecastHourNum']):
        for i, (X, y) in enumerate(pre_loader):
            wrf, obs = X
            label = y
            wrf = wrf.to(config_dict['Device'])
            obs = obs.to(config_dict['Device'])
            label = label.to(config_dict['Device'])
            pre_frames = model(wrf, obs)
            pre_frames = torch.sigmoid(pre_frames)
            pre_frames = pre_frames[:, :t+1]
            label = label[:, :t+1]

            eval_scores = calparams_epoch.cal_batch_sum(label[:, :], pre_frames[:, :])
            eval_scores_neighbor = calparams_epoch.cal_batch_neighbor(label[:, :], pre_frames[:, :])

            # eval_scores = calparams_epoch.cal_batch_sum(label[:, :6], pre_frames[:, :6])
            # eval_scores_neighbor = calparams_epoch.cal_batch_neighbor(label[:, :6], pre_frames[:, :6])
            # eval_scores = calparams_epoch.cal_batch_sum(label[:, 6:], pre_frames[:, 6:])
            # eval_scores_neighbor = calparams_epoch.cal_batch_neighbor(label[:, 6:], pre_frames[:, 6:])

            if torch.sum(label) > 0:
                vis.save_diff(pre_frames, label, i + 1)
                vis.save_oripre(pre_frames, i + 1)
            # output
            if i % 100 == 0:
                info = 'TEST INFO: ({}/{}) \n{}\n{}\n'.format(i + 1, len(pre_loader), str(eval_scores),
                                                              str(eval_scores_neighbor))
                print(info)

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
        print(info)
        print(POD)
        print(FAR)
        print(TS)
        print(ETS)

        print('neighbor')

        nPOD = nPOD + ' {:.5f}'.format(eval_scores_neighbor['POD'])
        nFAR = nFAR + ' {:.5f}'.format(eval_scores_neighbor['FAR'])
        nTS = nTS + ' {:.5f}'.format(eval_scores_neighbor['TS'])
        nETS = nETS + ' {:.5f}'.format(eval_scores_neighbor['ETS'])




if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'

    config_dict = read_config()

    print('pre = {}'.format(config_dict['NetName']))

    if not os.path.isdir(config_dict['VisResultFileDir']):
        os.makedirs(config_dict['VisResultFileDir'])

    DoPredict(config_dict)
