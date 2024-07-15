import torch
import torch.nn as nn
from torch.nn import functional as F
from Module import ASPP, RFB, h_swish
from backbone.PVTV2 import pvt_v2_b2
from torch.nn import Softmax

out_k = 64

class PSM(nn.Module):   
    def __init__(self, in_channel, out_channel):
        super(PSM, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
       
        self.refine_0 = nn.Sequential(nn.Conv2d(in_channel*2, out_channel, 3, 1, 1), nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True))
        self.refine_1 = nn.Sequential(nn.Conv2d(in_channel*2, out_channel, 3, 1, 1), nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True))
        self.refine_2 = nn.Sequential(nn.Conv2d(in_channel*2, out_channel, 3, 1, 1), nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True))
        
    def forward(self, u, v, rgb, last, up):
        temp = self.refine_0(torch.cat((u, v), dim=1))
        he = self.refine_1(torch.cat((temp, rgb), dim=1))
        if up == True:
            last = self.upsample_2(last)
        out = self.refine_2(torch.cat((last, he), dim=1)) 
        return out

class GCU_H(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(GCU_H,self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_channel, out_channels=in_channel//4, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_channel, out_channels=in_channel//4, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, flow, de_out):
        m_batchsize, _, height, width = de_out.size()
        proj_query = self.query_conv(de_out)
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
        proj_key = self.key_conv(flow)
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_value = self.value_conv(flow)
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)

        energy_H = (torch.bmm(proj_query_H, proj_key_H)).view(m_batchsize,width,height,height).permute(0,2,1,3)
        concate = self.softmax(energy_H)
        att_H = concate.permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)

        return out_H

class GCU_W(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(GCU_W,self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_channel, out_channels=in_channel//4, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_channel, out_channels=in_channel//4, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, flow, de_out):
        m_batchsize, _, height, width = de_out.size()
        proj_query = self.query_conv(de_out)
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)
        proj_key = self.key_conv(flow)
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        proj_value = self.value_conv(flow)
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width) 
        

        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)
        concate = self.softmax(energy_W)
        att_W = concate.contiguous().view(m_batchsize*height,width,width)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)
        
        return out_W
    
class CCU_W(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(CCU_W, self).__init__()
        self.relu = nn.ReLU(True)
        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.fuse = nn.Conv2d(in_channel*2, in_channel, 3, 1, 1)
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.refine = nn.Sequential(nn.Conv2d(in_channel, in_channel, 1, 1, 0), nn.BatchNorm2d(in_channel), h_swish())
        self.refine_1 = nn.Conv2d(in_channel, out_channel, 1, 1, 0)

    def forward(self, flow, de_out):
        fuse = self.fuse(torch.cat((flow, de_out), dim=1))
        att_w = self.pool_w(fuse)

        att_w = self.refine(att_w)
        att_w = self.refine_1(att_w).sigmoid()

        return flow * att_w

class CCU_H(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(CCU_H, self).__init__()
        self.relu = nn.ReLU(True)
        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.fuse = nn.Conv2d(in_channel*2, in_channel, 3, 1, 1)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))

        self.refine = nn.Sequential(nn.Conv2d(in_channel, in_channel, 1, 1, 0), nn.BatchNorm2d(in_channel), h_swish())
        self.refine_1 = nn.Conv2d(in_channel, out_channel, 1, 1, 0)

    def forward(self, flow, de_out):
        fuse = self.fuse(torch.cat((flow, de_out), dim=1))
        att_h = self.pool_h(fuse)

        att_h = self.refine(att_h)
        att_h = self.refine_1(att_h).sigmoid()

        return flow * att_h
    
class PPM_two(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(PPM_two, self).__init__()
        self.relu = nn.ReLU(True)
        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.refine = nn.Sequential(nn.Conv2d(in_channel*2, out_channel, 3, 1, 1), nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True))
        
    def forward(self, flow, de_out):
        flow = self.refine(torch.cat((flow, de_out), dim=1)) + flow
        
        return flow
    
    
class PPM_three(nn.Module):
    def __init__(self, in_channel, out_channel, mode, mode1):
        super(PPM_three, self).__init__()
        self.relu = nn.ReLU(True)
        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        if mode == 'H':
            self.refine_1 = CCU_H(in_channel, out_channel)
        else:
            self.refine_1 = CCU_W(in_channel, out_channel)

        if mode1 == 'H':
            self.refine_2 = GCU_H(in_channel, out_channel)
        else:
            self.refine_2 = GCU_W(in_channel, out_channel)

        self.fuse = nn.Sequential(nn.Conv2d(in_channel*3, out_channel, 3, 1, 1), nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True))

        self.agg = RFB(in_channel*2, out_channel)
        
    def forward(self, flow, de_out, last):
        last = self.upsample_2(last)
        de_out = self.upsample_2(de_out)
        flow_1 = self.refine_1(flow, de_out) + flow
        flow_2 = self.refine_2(flow, de_out) + flow
        enhance = self.fuse(torch.cat((flow_1, flow_2, flow), dim=1))
        
        out = self.agg(torch.cat((enhance, de_out), dim=1))

        return out

class out_block(nn.Module):
    def __init__(self, infilter):
        super(out_block, self).__init__()
        self.conv1 = nn.Sequential(
            *[nn.Conv2d(infilter, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True)])
        self.conv2 = nn.Conv2d(64, 1, 1)

    def forward(self, x, H, W):
        x = F.interpolate(self.conv1(x), (H, W), mode='bilinear', align_corners=True)
        return self.conv2(x)
    
class PANet(nn.Module):
    def __init__(self, backbone_u, backbone_v, backbone_rgb):
        super(PANet, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        
        self.backbone_u = backbone_u
        self.backbone_v = backbone_v
        self.backbone_rgb = backbone_rgb

        compression_u = []
        compression_u.append(nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), self.relu,
                                           nn.Conv2d(64, out_k, 1), nn.BatchNorm2d(out_k), self.relu))
        compression_u.append(nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), self.relu,
                                           nn.Conv2d(128, out_k, 1), nn.BatchNorm2d(out_k), self.relu))
        compression_u.append(nn.Sequential(nn.Conv2d(320, 320, 3, 1, 1), nn.BatchNorm2d(320), self.relu,
                                           nn.Conv2d(320, out_k, 1), nn.BatchNorm2d(out_k), self.relu))
        compression_u.append(nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), self.relu,
                                           nn.Conv2d(512, out_k, 1), nn.BatchNorm2d(out_k), self.relu))
        self.cp_u = nn.ModuleList(compression_u)

        compression_v = []
        compression_v.append(nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), self.relu,
                                           nn.Conv2d(64, out_k, 1), nn.BatchNorm2d(out_k), self.relu))
        compression_v.append(nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), self.relu,
                                           nn.Conv2d(128, out_k, 1), nn.BatchNorm2d(out_k), self.relu))
        compression_v.append(nn.Sequential(nn.Conv2d(320, 320, 3, 1, 1), nn.BatchNorm2d(320), self.relu,
                                           nn.Conv2d(320, out_k, 1), nn.BatchNorm2d(out_k), self.relu))
        compression_v.append(nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), self.relu,
                                           nn.Conv2d(512, out_k, 1), nn.BatchNorm2d(out_k), self.relu))
        self.cp_v = nn.ModuleList(compression_v)
        
        compression_rgb = []
        compression_rgb.append(nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), self.relu,
                                           nn.Conv2d(64, out_k, 1), nn.BatchNorm2d(out_k), self.relu))
        compression_rgb.append(nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), self.relu,
                                           nn.Conv2d(128, out_k, 1), nn.BatchNorm2d(out_k), self.relu))
        compression_rgb.append(nn.Sequential(nn.Conv2d(320, 320, 3, 1, 1), nn.BatchNorm2d(320), self.relu,
                                           nn.Conv2d(320, out_k, 1), nn.BatchNorm2d(out_k), self.relu))
        compression_rgb.append(nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), self.relu,
                                           nn.Conv2d(512, out_k, 1), nn.BatchNorm2d(out_k), self.relu))
        self.cp_rgb = nn.ModuleList(compression_rgb)

        self.aspp = ASPP(512, out_k, rates = [1, 6, 12, 18])

        self.PSM_0 = PSM(out_k, out_k)
        self.PSM_1 = PSM(out_k, out_k)
        self.PSM_2 = PSM(out_k, out_k)
        self.PSM_3 = PSM(out_k, out_k)
        
        self.PPM_u_0 = PPM_three(out_k, out_k, 'H', 'H')
        self.PPM_u_1 = PPM_three(out_k, out_k, 'H', 'H')
        self.PPM_u_2 = PPM_three(out_k, out_k, 'H', 'H')
        self.PPM_u_3 = PPM_two(out_k, out_k)

        self.PPM_v_0 = PPM_three(out_k, out_k, 'W', 'W')
        self.PPM_v_1 = PPM_three(out_k, out_k, 'W', 'W')
        self.PPM_v_2 = PPM_three(out_k, out_k, 'W', 'W')
        self.PPM_v_3 = PPM_two(out_k, out_k)

        self.pred_head_u = out_block(out_k)
        self.pred_head_v = out_block(out_k)
        self.pred_head_rgb = out_block(out_k)

    def forward(self, up, left, rgb):
        #[64,64,64] -> [128,32,32] -> [320,16,16] -> [512,8,8]
        u_list= self.backbone_u(up)              
        v_list= self.backbone_v(left)
        rgb_list= self.backbone_rgb(rgb)      
        rgb_aspp = self.aspp(rgb_list[-1])
        
        for i in range(4):
            rgb_list[i] = self.cp_rgb[i](rgb_list[i])
            u_list[i] = self.cp_u[i](u_list[i])
            v_list[i] = self.cp_v[i](v_list[i])

        eput_u_3 = self.PPM_u_3(u_list[3], rgb_aspp)
        eput_v_3 = self.PPM_v_3(v_list[3], rgb_aspp)
        dput_3 = self.PSM_3(eput_u_3, eput_v_3, rgb_list[3], rgb_aspp, up = False)

        eput_u_2 = self.PPM_u_2(u_list[2], dput_3, eput_u_3)
        eput_v_2 = self.PPM_v_2(v_list[2], dput_3, eput_v_3)
        dput_2 = self.PSM_2(eput_u_2, eput_v_2, rgb_list[2], dput_3, up = True)

        eput_u_1 = self.PPM_u_1(u_list[1], dput_2, eput_u_2)
        eput_v_1 = self.PPM_v_1(v_list[1], dput_2, eput_v_2)
        dput_1 = self.PSM_1(eput_u_1, eput_v_1, rgb_list[1], dput_2, up = True)

        eput_u_0 = self.PPM_u_0(u_list[0], dput_1, eput_u_1)
        eput_v_0 = self.PPM_v_0(v_list[0], dput_1, eput_v_1)
        dput_0 = self.PSM_0(eput_u_0, eput_v_0, rgb_list[0], dput_1, up = True)

        pre_rgb = self.pred_head_rgb(dput_0, 256, 256)
        pre_u = self.pred_head_u(eput_u_0, 256, 256)
        pre_v = self.pred_head_v(eput_v_0, 256, 256)
        
        return pre_rgb, pre_u, pre_v
        
def build_PANet():
    model_u = pvt_v2_b2()
    model_v = pvt_v2_b2()
    model_rgb = pvt_v2_b2()
    return PANet(model_u, model_v, model_rgb)
 
 