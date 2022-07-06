import time
import pywt
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd import Variable, gradcheck

class DWT_Function(Function):
    @staticmethod
    def forward(ctx, x, w_ll, w_lh, w_hl, w_hh):
        x = x.contiguous()
        ctx.save_for_backward(w_ll, w_lh, w_hl, w_hh)
        ctx.shape = x.shape

        dim = x.shape[1]
        x_ll = torch.nn.functional.conv2d(x, w_ll.expand(dim, -1, -1, -1), stride = 2, groups = dim)
        x_lh = torch.nn.functional.conv2d(x, w_lh.expand(dim, -1, -1, -1), stride = 2, groups = dim)
        x_hl = torch.nn.functional.conv2d(x, w_hl.expand(dim, -1, -1, -1), stride = 2, groups = dim)
        x_hh = torch.nn.functional.conv2d(x, w_hh.expand(dim, -1, -1, -1), stride = 2, groups = dim)
        x = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            w_ll, w_lh, w_hl, w_hh = ctx.saved_tensors
            B, C, H, W = ctx.shape
            dx = dx.view(B, 4, -1, H//2, W//2)

            dx = dx.transpose(1,2).reshape(B, -1, H//2, W//2)
            filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0).repeat(C, 1, 1, 1)
            dx = torch.nn.functional.conv_transpose2d(dx, filters, stride=2, groups=C)

        return dx, None, None, None, None

class IDWT_Function(Function):
    @staticmethod
    def forward(ctx, x, filters):
        ctx.save_for_backward(filters)
        ctx.shape = x.shape

        B, _, H, W = x.shape
        x = x.view(B, 4, -1, H, W).transpose(1, 2)
        C = x.shape[1]
        x = x.reshape(B, -1, H, W)
        filters = filters.repeat(C, 1, 1, 1)
        x = torch.nn.functional.conv_transpose2d(x, filters, stride=2, groups=C)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            filters = ctx.saved_tensors
            filters = filters[0]
            B, C, H, W = ctx.shape
            C = C // 4
            dx = dx.contiguous()

            w_ll, w_lh, w_hl, w_hh = torch.unbind(filters, dim=0)
            x_ll = torch.nn.functional.conv2d(dx, w_ll.unsqueeze(1).expand(C, -1, -1, -1), stride = 2, groups = C)
            x_lh = torch.nn.functional.conv2d(dx, w_lh.unsqueeze(1).expand(C, -1, -1, -1), stride = 2, groups = C)
            x_hl = torch.nn.functional.conv2d(dx, w_hl.unsqueeze(1).expand(C, -1, -1, -1), stride = 2, groups = C)
            x_hh = torch.nn.functional.conv2d(dx, w_hh.unsqueeze(1).expand(C, -1, -1, -1), stride = 2, groups = C)
            dx = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return dx, None

class IDWT_2D(nn.Module):
    def __init__(self, wave):
        super(IDWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        rec_hi = torch.Tensor(w.rec_hi)
        rec_lo = torch.Tensor(w.rec_lo)
        
        w_ll = rec_lo.unsqueeze(0)*rec_lo.unsqueeze(1)
        w_lh = rec_lo.unsqueeze(0)*rec_hi.unsqueeze(1)
        w_hl = rec_hi.unsqueeze(0)*rec_lo.unsqueeze(1)
        w_hh = rec_hi.unsqueeze(0)*rec_hi.unsqueeze(1)

        w_ll = w_ll.unsqueeze(0).unsqueeze(1)
        w_lh = w_lh.unsqueeze(0).unsqueeze(1)
        w_hl = w_hl.unsqueeze(0).unsqueeze(1)
        w_hh = w_hh.unsqueeze(0).unsqueeze(1)
        filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0)
        self.register_buffer('filters', filters)
        self.filters = self.filters.to(dtype=torch.float16)

    def forward(self, x):
        return IDWT_Function.apply(x, self.filters)

class DWT_2D(nn.Module):
    def __init__(self, wave):
        super(DWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        dec_hi = torch.Tensor(w.dec_hi[::-1]) 
        dec_lo = torch.Tensor(w.dec_lo[::-1])

        w_ll = dec_lo.unsqueeze(0)*dec_lo.unsqueeze(1)
        w_lh = dec_lo.unsqueeze(0)*dec_hi.unsqueeze(1)
        w_hl = dec_hi.unsqueeze(0)*dec_lo.unsqueeze(1)
        w_hh = dec_hi.unsqueeze(0)*dec_hi.unsqueeze(1)

        self.register_buffer('w_ll', w_ll.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_lh', w_lh.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hl', w_hl.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hh', w_hh.unsqueeze(0).unsqueeze(0))

        self.w_ll = self.w_ll.to(dtype=torch.float16)
        self.w_lh = self.w_lh.to(dtype=torch.float16)
        self.w_hl = self.w_hl.to(dtype=torch.float16)
        self.w_hh = self.w_hh.to(dtype=torch.float16)

    def forward(self, x):
        return DWT_Function.apply(x, self.w_ll, self.w_lh, self.w_hl, self.w_hh)
        
'''
def test_time(x, dwt1, dwt2):
    loop = 1000
    total_time1 = 0
    total_time2 = 0

    for i in range(loop):
        start = time.time()
        y1 = dwt1(x)
        torch.cuda.synchronize()
        end = time.time()
        total_time1 += end - start
    
    for i in range(loop):
        start = time.time()
        y2_ll, YH = dwt2(x)
        torch.cuda.synchronize()
        end = time.time()
        total_time2 += end - start

    print(total_time1)
    print(total_time2)

def test_diff(x, dwt1, dwt2):
    y1 = dwt1(x)
    B, C, H, W = y1.shape
    y1 = y1.view(B, 4, -1, H, W)
    y1_ll = y1[:, 0] 
    y1_lh = y1[:, 1]
    y1_hl = y1[:, 2]
    y1_hh = y1[:, 3]
    y2_ll, YH = dwt2(x)
    y2_lh = YH[0][:,:,0]
    y2_hl = YH[0][:,:,1]
    y2_hh = YH[0][:,:,2]
    diff1 = (y1_ll - y2_ll).max()
    diff2 = (y1_lh - y2_lh).max()
    diff3 = (y1_hl - y2_hl).max()
    diff4 = (y1_hh - y2_hh).max()
    print(diff1)
    print(diff2)
    print(diff3)
    print(diff4)

def test_idfiff(x, idwt1, idwt2):
    y1 = idwt1(x)

    x = x.view(x.size(0), 4, -1, x.size(-2), x.size(-1))
    y2 = idwt2((x[:, 0], [x[:,1:].transpose(1, 2)]))
    diff = (y1-y2).max()
    print(diff)

def test_itime(x, idwt1, idwt2):
    loop = 1000
    total_time1 = 0
    total_time2 = 0

    for i in range(loop):
        start = time.time()
        y1 = idwt1(x)
        torch.cuda.synchronize()
        end = time.time()
        total_time1 += end - start
    
    for i in range(loop):
        start = time.time()
        x = x.view(x.size(0), 4, -1, x.size(-2), x.size(-1))
        y2 = idwt2((x[:, 0], [x[:,1:].transpose(1, 2)]))
        torch.cuda.synchronize()
        end = time.time()
        total_time2 += end - start

    print(total_time1)
    print(total_time2)

if __name__ == '__main__':
    #size = (96, 32, 56, 56)
    #size = (96, 64, 28, 28)
    size = (96, 160, 14, 14)
    x = torch.randn(size).cuda().to(dtype=torch.float16)
    dwt1 = DWT_2D('haar').cuda()
    dwt2 = DWTForward(wave='haar').cuda()
    test_diff(x, dwt1, dwt2)
    test_time(x, dwt1, dwt2)

    #size = (96, 32*4, 28, 28)
    #size = (96, 64*4, 14, 14)
    #size = (96, 160*4, 7, 7)
    #x = torch.randn(size).cuda().to(dtype=torch.float16)
    #idwt1 = IDWT_2D('haar').cuda()
    #idwt2 = DWTInverse(wave='haar').cuda()
    #test_idfiff(x, idwt1, idwt2)
    #test_itime(x, idwt1, idwt2)

def test_dwt_grad():
    size = (4, 8, 14, 14)
    x = torch.randn(size).double()

    w = pywt.Wavelet('haar')
    dec_hi = torch.Tensor(w.dec_hi[::-1]) 
    dec_lo = torch.Tensor(w.dec_lo[::-1])

    w_ll = (dec_lo.unsqueeze(0)*dec_lo.unsqueeze(1)).unsqueeze(0).unsqueeze(0).double()
    w_lh = (dec_lo.unsqueeze(0)*dec_hi.unsqueeze(1)).unsqueeze(0).unsqueeze(0).double()
    w_hl = (dec_hi.unsqueeze(0)*dec_lo.unsqueeze(1)).unsqueeze(0).unsqueeze(0).double()
    w_hh = (dec_hi.unsqueeze(0)*dec_hi.unsqueeze(1)).unsqueeze(0).unsqueeze(0).double()

    input = (
        Variable(x, requires_grad=True),
        Variable(w_ll, requires_grad=False),
        Variable(w_lh, requires_grad=False),
        Variable(w_hl, requires_grad=False),
        Variable(w_hh, requires_grad=False),
    )
    test = gradcheck(DWT_Function.apply, input)
    print("test:", test)

def test_idwt_grad():
    size = (4, 2*8, 7, 7)
    x = torch.randn(size).double()

    w = pywt.Wavelet('haar')
    rec_hi = torch.Tensor(w.rec_hi)
    rec_lo = torch.Tensor(w.rec_lo)
        
    w_ll = rec_lo.unsqueeze(0)*rec_lo.unsqueeze(1)
    w_lh = rec_lo.unsqueeze(0)*rec_hi.unsqueeze(1)
    w_hl = rec_hi.unsqueeze(0)*rec_lo.unsqueeze(1)
    w_hh = rec_hi.unsqueeze(0)*rec_hi.unsqueeze(1)

    w_ll = w_ll.unsqueeze(0).unsqueeze(1)
    w_lh = w_lh.unsqueeze(0).unsqueeze(1)
    w_hl = w_hl.unsqueeze(0).unsqueeze(1)
    w_hh = w_hh.unsqueeze(0).unsqueeze(1)
    filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0).double()

    input = (
        Variable(x, requires_grad=True),
        Variable(filters, requires_grad=False),
    )
    test = gradcheck(IDWT_Function.apply, input)
    print("test:", test)

if __name__ == "__main__":
    test_dwt_grad()
'''
