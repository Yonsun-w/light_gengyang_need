import torch
import torch.nn as nn
from layers.ConvLSTM import ConvLSTM2D
from layers.moe_ADSNet import ADSNet_Model
from layers.moe_LightNet import LightNet_Model


class Attention_model(nn.Module):
    # obs = 1 h = 64
    def __init__(self, obs_channels, h_channels):
        super(Attention_model, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(obs_channels, h_channels, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4)
        )
        self.att_weight1 = nn.Sequential(
            nn.Conv2d(h_channels, h_channels, kernel_size=5, stride=1, padding=2)
        )
        self.att_weight2 = nn.Sequential(
            nn.Conv2d(h_channels, h_channels, kernel_size=5, stride=1, padding=2)
        )
        self.CNN_hw = nn.Conv2d(128 * 2, 1, kernel_size=1, stride=1)

      # h_old 64  39 39      obs 1  159 159
    def forward(self, a_h_old, l_h_old, obs):

        # 对obs进行放缩 也变成64 39 39
        obs = self.conv1(obs)

        # 对a_h l_h进行采样
        a_h_old = self.att_weight1(a_h_old)
        l_h_old = self.att_weight2(l_h_old)

        # obs 分别乘 a_h_old
        a_h_old = torch.mul(obs, a_h_old).unsqueeze(dim=0)
        l_h_old = torch.mul(obs, l_h_old).unsqueeze(dim=0)

        # 专家数 64 39 39 分别代表每个专家每个点的权重
        res = torch.cat([a_h_old, l_h_old], dim=0)
        res = torch.softmax(res, dim=0)

        return res



class Encoder_wrf_model(nn.Module):
    def __init__(self, tra_frames, channels, row_col):
        super(Encoder_wrf_model, self).__init__()
        self.tra_frames = tra_frames
        self.wrf_encoder_conv2d = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.wrf_encoder_convLSTM2D = ConvLSTM2D(64, 32, kernel_size=5, img_rowcol=(row_col//2)//2)

    def forward(self, wrf):
        # wrf : [frames, batch_size, channels, x, y]
        batch_size = wrf.shape[1]
        wrf_conv = [None] * self.tra_frames
        for i in range(self.tra_frames):
            wrf_conv[i] = self.wrf_encoder_conv2d(wrf[i])
            # wrf[i]=torch.Size([4, 29, 159, 159]),wrf_conv[i]=torch.Size([4, 64, 39, 39])
            # print('wrf[i]={},wrf_conv[i]={}'.format(wrf[i].shape, wrf_conv[i].shape))
        wrf_h = torch.zeros([batch_size, 32, wrf_conv[0].shape[2], wrf_conv[0].shape[3]], dtype=torch.float32).to(wrf.device)
        wrf_c = torch.zeros([batch_size, 32, wrf_conv[0].shape[2], wrf_conv[0].shape[3]], dtype=torch.float32).to(wrf.device)

        for i in range(self.tra_frames):
            wrf_h, wrf_c = self.wrf_encoder_convLSTM2D(wrf_conv[i], wrf_h, wrf_c)

        return wrf_h, wrf_c

class Encoder_obs_model(nn.Module):
    def __init__(self, tra_frames, channels, row_col):
        super(Encoder_obs_model, self).__init__()
        self.tra_frames = tra_frames
        self.obs_encoder_conv2d = nn.Sequential(
                nn.Conv2d(channels, 4, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(4, 4, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
        self.obs_encoder_convLSTM2D = ConvLSTM2D(4, 8, kernel_size=5, img_rowcol=(row_col//2)//2)

    def forward(self, obs):
        # obs : [frames, batch_size, channels, x, y]
        batch_size = obs.shape[1]
        obs_conv = [None] * self.tra_frames
        for i in range(self.tra_frames):
            obs_conv[i] = self.obs_encoder_conv2d(obs[i])
            # obs_conv[i].shape=torch.Size([4, 4, 39, 39])

        obs_h = torch.zeros([batch_size, 8, obs_conv[0].shape[2], obs_conv[0].shape[3]], dtype=torch.float32).to(obs.device)
        obs_c = torch.zeros([batch_size, 8, obs_conv[0].shape[2], obs_conv[0].shape[3]], dtype=torch.float32).to(obs.device)

        for i in range(self.tra_frames):
            obs_h, obs_c = self.obs_encoder_convLSTM2D(obs_conv[i], obs_h, obs_c)

        return obs_h, obs_c



class MOE_Model(nn.Module):
     # obs_tra_frames = TruthHistoryHourNum   wrf_tra_frames=config_dict['ForecastHourNum']
     #  wrf_channels=config_dict['WRFChannelNum'],
    def __init__(self, truth_history_hour_num,forecast_hour_num, row_col,wrf_channels, obs_channel, ads_net_model_path = 'MOE/moe_ads_model_maxETS.pkl',
                 light_net_model_path='MOE/light_model_maxETS.pkl'):
        super(MOE_Model, self).__init__()

        self.truth_history_hour_num = truth_history_hour_num
        self.forecast_hour_num = forecast_hour_num




        ADSNet_expert = ADSNet_Model(obs_tra_frames=truth_history_hour_num, obs_channels=obs_channel,
                             wrf_tra_frames=forecast_hour_num,
                             wrf_channels=wrf_channels, row_col=row_col).to('cuda')

        LightNet_expert = LightNet_Model(obs_tra_frames=truth_history_hour_num, obs_channels=obs_channel,
                             wrf_tra_frames=forecast_hour_num,
                             wrf_channels=wrf_channels, row_col=row_col).to('cuda')


        # 这两个专家是为了获取真实时间输出的 他和上两个不同的地方在于 预测时间不一样
        ADSNet_expert_old = ADSNet_Model(obs_tra_frames=truth_history_hour_num, obs_channels=obs_channel,
                             wrf_tra_frames=forecast_hour_num,
                             wrf_channels=wrf_channels, row_col=row_col).to('cuda')

        LightNet_expert_old = LightNet_Model(obs_tra_frames=truth_history_hour_num, obs_channels=obs_channel,
                             wrf_tra_frames=forecast_hour_num,
                             wrf_channels=wrf_channels, row_col=row_col).to('cuda')

        ads_model_file = torch.load(ads_net_model_path, map_location=torch.device('cuda'))
        ADSNet_expert.load_state_dict(ads_model_file)
        light_model_file = torch.load(light_net_model_path, map_location=torch.device('cuda'))
        LightNet_expert.load_state_dict(light_model_file)
        self.expert_list = [ADSNet_expert, LightNet_expert]
        # 这个只是单独的为了获取训练的值，其实应该放在读取数据截断
        self.out_put = [ADSNet_expert_old, LightNet_expert_old]
        self.conv1 = nn.Sequential(
            nn.Conv2d(128, 64, 1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(24, 64, 1)
        )
        self.CNN_module3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1, stride=1)
        )
        self.attention_expert = nn.ModuleList([Attention_model(obs_channels=1, h_channels=64).to('cuda') for i in range(self.forecast_hour_num)])

         # 前两个参数是当前的输入 为了预测之后的, 中间两个是为了获取在当前时间到预测时间长度之前的专家模拟输入
         # 最后一个参数是当前到预测长度之前的真实输入
    def forward(self, wrf, obs, wrf_old, obs_old, obs_forces_old):

        batch_size = wrf.size(0)
        # 首先得到不同专家模型的结果
        ads_pre, ads_pre_h = self.expert_list[0](wrf, obs)
        light_pre, light_pre_h = self.expert_list[1](wrf, obs)

        ads_old, ads_old_h = self.out_put[0](wrf_old, obs_old)
        light_old, light_old_h = self.out_put[1](wrf_old, obs_old)


        # 然后我们获取到了ads_h, light_h 这时候让他们经过归一化 这里的卷积和下边的卷积是相同的，为了保证获取到的处理都是一致的特征
        # 这样才能保证obs关注到的点

        # 初始化权重注意力
        att_weight = torch.ones(2, batch_size, 64, 39, 39).cuda() * 0.5

        obs_forces_old = obs_forces_old.permute(1, 0, 4, 2, 3)

        weights = [None] * self.forecast_hour_num
        # 首先获得了所有实测的点，让他们进入一个注意力编码层
        for i in range(self.forecast_hour_num):
            a_h_o = ads_old_h[i:i+1]
            l_h_o = light_old_h[i:i+1]
            # 卷积归一然后解码 4,64,39,39
            l_h_o = l_h_o.squeeze(dim=0)
            a_h_o = a_h_o.squeeze(dim=0)
            a_h_o = self.conv1(a_h_o) * att_weight[0].squeeze(dim=0)
            l_h_o = self.conv2(l_h_o) * att_weight[1].squeeze(dim=0)
            # 将他们和真实数据送入一个注意力层 得到一个权重分布 该权重分布形状应该是 专家数量 batch channel h w 这种
            # 代表每个点的权重分布
            att_weight = self.attention_expert[i](a_h_o, l_h_o, obs_forces_old[i])
            weights[i] = att_weight

        res = [None] * self.forecast_hour_num

        att_weight_pre = torch.ones(2, batch_size, 64, 39, 39).cuda() * 0.5

        for i in range(self.forecast_hour_num):
            #  4, 128, 39, 39
            a_h = ads_pre_h[i:i+1].squeeze(dim=0)
            #  4, 24, 39, 39
            l_h = light_pre_h[i:i+1].squeeze(dim=0)
            # 卷积归一然后解码
            a_h = self.conv1(a_h)
            l_h = self.conv2(l_h)
            att_weight_pre = weights[i] * att_weight_pre
            att_weight_pre = torch.softmax(att_weight, dim=0)
            ads_w = att_weight_pre[0].squeeze(dim=0)
            light_w = att_weight_pre[1].squeeze(dim=0)
            fusion = a_h * ads_w + l_h * light_w
            fusion = self.CNN_module3(fusion)
            # 此时得到的是真是的
            res[i] = fusion

        res = torch.stack(res, dim=0)

        #  batch hour  1  159 159
        res = res.permute(1, 0, 3, 4, 2)


        return res
