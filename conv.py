import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight.data, gain=init.calculate_gain('relu'))
        # init.xavier_uniform_(m.bias.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=0.2)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


class Conv4D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, padding, stride, is_bias=True):
        super(Conv4D, self).__init__()
        conv = torch.zeros(out_channels, kernel[0] * kernel[1] * kernel[2] * kernel[3], 1).repeat(1, 1, in_channels)
        conv = conv.reshape(out_channels, kernel[0] * kernel[1] * kernel[2] * kernel[3] * in_channels)
        self.conv = torch.nn.Parameter(conv)
        if is_bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_channels))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = kernel
        self.padding = padding
        self.stride = stride
        self.is_bias = is_bias

    def forward(self, x):
        batch_size, in_channels, u, v, h, w = x.shape
        assert in_channels == self.in_channels, 'in_channels does not correspond'
        x = F.pad(x, (self.padding[3], self.padding[3],
                      self.padding[2], self.padding[2],
                      self.padding[1], self.padding[1],
                      self.padding[0], self.padding[0]))
        for i in range(4):
            x = x.unfold(i + 2, self.kernel[i], self.stride[i])
        x = x.permute(0, 6, 7, 8, 9, 1, 2, 3, 4, 5)
        x = x.reshape(batch_size, -1, x.shape[6] * x.shape[7] * x.shape[8] * x.shape[9])
        x = torch.matmul(self.conv, x)
        if self.is_bias:
            x += self.bias.unsqueeze(0).unsqueeze(-1)
        x = x.reshape(batch_size, self.out_channels, u, v, h, w)
        return x


class ConvXD(nn.Module):
    def __init__(self, dimension, in_channels, out_channels, kernel, padding, stride, is_bias=True):
        super(ConvXD, self).__init__()
        # if len(kernel) == 1:
        #     kernel = [kernel for _ in range(dimension)]
        # if len(padding) == 1:
        #     padding = [padding for _ in range(dimension)]
        # if len(stride) == 1:
        #     stride = [stride for _ in range(dimension)]
        kernel_mul = 1
        for i in range(dimension):
            kernel_mul *= kernel[i]
        weight = torch.zeros(out_channels, kernel_mul, 1).repeat(1, 1, in_channels)
        weight = weight.reshape(out_channels, kernel_mul * in_channels)
        self.weight = torch.nn.Parameter(weight)
        if is_bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_channels))
        self.dimension = dimension
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = kernel
        self.padding = padding
        self.stride = stride
        self.is_bias = is_bias

    def forward(self, x):
        batch_size, in_channels = x.shape[:2]
        x_shape = x.shape[2:]
        assert in_channels == self.in_channels, 'in_channels does not correspond'
        pad = []
        for i in range(self.dimension):
            pad.append(self.padding[self.dimension - i - 1])
            pad.append(self.padding[self.dimension - i - 1])
        x = F.pad(x, pad)
        for i in range(self.dimension):
            x = x.unfold(i + 2, self.kernel[i], self.stride[i])
        transpose_arr = [i for i in range(2 * self.dimension + 2)]
        for i in range(self.dimension + 1):
            tmp = transpose_arr[1]
            transpose_arr.pop(1)
            transpose_arr.append(tmp)
        x = x.permute(*transpose_arr)
        x_mul = 1
        for i in range(self.dimension):
            x_mul *= x.shape[-1 - i]
        x = x.reshape(batch_size, -1, x_mul)
        x = torch.matmul(self.weight, x)
        if self.is_bias:
            x += self.bias.unsqueeze(0).unsqueeze(-1)
        x = x.reshape(batch_size, self.out_channels, *x_shape)
        return x


if __name__ == '__main__':

    model = ConvXD(dimension=4, in_channels=3, out_channels=6, kernel=(1, 3, 3, 3), padding=(0, 1, 1, 1),
                   stride=(1, 1, 1, 1), is_bias=True)

    model.apply(weights_init_xavier)

    x = torch.arange(0, 1125).resize_(1, 3, 5, 3, 5, 5).float()
    label = torch.ones(1, 6, 5, 3, 5, 5).float()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion_train = torch.nn.L1Loss()

    for i in range(1000):
        optimizer.zero_grad()
        y = model(x)
        loss = criterion_train(y, label)
        print(loss.item())
        loss.backward()
        optimizer.step()

    print(model(x))
