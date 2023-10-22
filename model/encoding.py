import torch
import torch.nn as nn

class PositionalEncoding2D(nn.Module):
  def __init__(self, dim, h_max, w_max):
    super(PositionalEncoding2D, self).__init__()
    self.h_max = h_max
    self.max_w = w_max
    self.dim = dim
    self.pe = torch.zeros((1, dim, h_max, w_max), device=torch.device(
      "cuda" if torch.cuda.is_available() else "cpu"), requires_grad=False)

    div = torch.exp(-torch.arange(0., dim // 2, 2) / dim *
                    torch.log(torch.tensor(10000.0))).unsqueeze(1)
    w_pos = torch.arange(0., w_max)
    h_pos = torch.arange(0., h_max)
    self.pe[:, :dim // 2:2, :, :] = torch.sin(
      h_pos * div).unsqueeze(0).unsqueeze(3).repeat(1, 1, 1, w_max)
    self.pe[:, 1:dim // 2:2, :, :] = torch.cos(
      h_pos * div).unsqueeze(0).unsqueeze(3).repeat(1, 1, 1, w_max)
    self.pe[:, dim // 2::2, :, :] = torch.sin(
      w_pos * div).unsqueeze(0).unsqueeze(2).repeat(1, 1, h_max, 1)
    self.pe[:, dim // 2 + 1::2, :, :] = torch.cos(
      w_pos * div).unsqueeze(0).unsqueeze(2).repeat(1, 1, h_max, 1)

  def forward(self, x):
    """
    Add 2D positional encoding to x
    x: (B, C, H, W)
    """
    return x + self.pe[:, :, :x.size(2), :x.size(3)]

  def get_pe_by_size(self, h, w, device):
    return self.pe[:, :, :h, :w].to(device)
