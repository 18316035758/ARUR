def prep_clff(y_true, y_pred, threshold):
    threshold = torch.tensor(threshold)
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    y_true_bin = torch.where(y_true >= threshold, 1, 0)     # 将实际值和预测值转换为二进制
    y_pred_bin = torch.where(y_pred >= threshold, 1, 0)
    hits = torch.sum((y_true_bin == 1) & (y_pred_bin == 1))
    misses = torch.sum((y_true_bin == 1) & (y_pred_bin == 0))
    falsealarms = torch.sum((y_true_bin == 0) & (y_pred_bin == 1))
    correctnegatives = torch.sum((y_true_bin == 0) & (y_pred_bin == 0))
    return hits, misses, falsealarms, correctnegatives


def assess2(y_true, y_pred, threshold):
    global CSI, BIAS, FAR, POD
    B, T, h, w = y_true.shape
    ssim_all, csi_all, bias_all, far_all, pod_all = 0.0, 0.0, 0.0, 0.0, 0.0
    csi = []
    far = []
    pod = []
    for b in range(B):
        for t in range(T):
            y_pree = y_pred[b, t]
            y_truee = y_true[b, t]
            ssim_all = ssim_all + ssim(y_truee, y_pree)
            hits, misses, falsealarms, correctnegatives = prep_clff(y_truee, y_pree, threshold=threshold)
            csi_all = csi_all+(hits/(hits + falsealarms + misses))
            far_all = far_all+(falsealarms / (hits + falsealarms))
            pod_all = pod_all+(hits / (hits + misses))
            csi.append(csi_all.cpu().detach().numpy())
            far.append(far_all.cpu().detach().numpy())
            pod.append(pod_all.cpu().detach().numpy())
            CSI = pd.DataFrame(csi)
            CSI.columns = ['csi']
            FAR = pd.DataFrame(far)
            FAR.columns = ['far']
            POD = pd.DataFrame(pod)
            POD.columns = ['pod']
    recode = pd.concat([CSI['csi'], FAR['far'], POD['pod']], axis=1)
    return ssim_all / (B * T), csi_all / (B * T), far_all / (B * T), pod_all / (B * T), recode


class RadarCell(nn.Module):
    def __init__(self, sort):
        super(RadarCell, self).__init__()
        self.sort = sort + 1
        self.wb = nn.Parameter(torch.randn(self.sort)).to(device)
        self.dataup = Dataup().to(device)
        self.relu = nn.ReLU()

    def forward(self, x_true, rh, div, wb_test, test=True):
        if test:
            wb = wb_test
        else:
            wb = self.wb
        x_true = x_true.to(device)
        div = div.to(device)
        rh = rh.to(device)
        B, T, h, w = x_true.shape
        cell_fin = torch.ones(B, T, h, w).to(device)
        for b in range(B):
            for t in range(T):
                x_truee = x_true[b, t]
                rh_input = rh[b, t]
                div_input = div[b, t]
                rh_radars = self.dataup(rh_input)    # rmaps to radar
                div_radars = self.dataup(div_input)
                cell_out = self.relu(x_truee + wb[0]*rh_radars + wb[1]*div_radars + wb[-1]).to(device)
                cell_fin[b, t, :, :] = cell_out
        return cell_fin, wb


class SELayer(nn.Module):
    def __init__(self, channel, redio=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channel, channel//redio), nn.ReLU(inplace=True),
                                nn.Linear(channel//redio, channel // (redio*2)), nn.ReLU(inplace=True),
                                nn.Linear(channel//(redio*2), channel), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c).to(device)   # b,c,h,w->b,c,1,1->b,c
        y = self.fc(y).view(b, c, 1, 1).to(device)   # b,c->b,c/16->b,c,1,1
        return x * y


class UNet(nn.Module):
    def __init__(self, in_channel, out_channel, fin_channel, count):   # count是雷达和物理量的个数
        super(UNet, self).__init__()
        self.c1 = ConvBlock(in_channel, out_channel)
        self.d1 = DownSample(out_channel)
        self.c2 = ConvBlock(out_channel, out_channel * 2)
        self.d2 = DownSample(out_channel * 2)
        self.c3 = ConvBlock(out_channel * 2, out_channel * 4)
        self.d3 = DownSample(out_channel * 4)
        self.c4 = ConvBlock(out_channel * 4, out_channel * 8)
        self.u1 = UpSample(out_channel * 8 * count)
        self.c6 = ConvBlock(out_channel * 8 * count, out_channel * 4)
        self.u2 = UpSample(out_channel * 4)
        self.c7 = ConvBlock(out_channel * 4 * count, out_channel * 2)
        self.u3 = UpSample(out_channel * 2)
        self.c8 = ConvBlock(out_channel * 2 * count, out_channel)
        self.out = nn.Conv2d(out_channel, fin_channel, 3, 1, 1)
        self.relu = nn.ReLU()
        self.se4 = SELayer(out_channel * 8)

    def forward(self, x):
        R1 = self.c1(x)
        R2 = self.c2(self.d1(R1))
        R3 = self.c3(self.d2(R2))
        R4 = self.c4(self.d3(R3))
        SE4 = self.se4(R4)
        O1 = self.c6(self.u1(SE4, R3, bilinear=True))
        O2 = self.c7(self.u2(O1, R2, bilinear=True))
        O3 = self.c8(self.u3(O2, R1, bilinear=True))
        out = self.relu(self.out(O3))
        return out


class All(nn.Module):
    def __init__(self):
        super(All, self).__init__()
        self.radar_combine = RadarCell(sort=2).to(device)
        self.unet = UNet(10, 64, 10, 1).to(device)

    def forward(self, x0, x2, x3, wb_out1, tests=True):
        if tests:
            x1_x2train, wb_out11 = self.radar_combine(x0, x2, x3, wb_out1, test=True)
        else:
            x1_x2train, wb_out11 = self.radar_combine(x0, x2, x3, wb_out1, test=False)
        outputs = self.unet(x1_x2train)
        return outputs, wb_out11
