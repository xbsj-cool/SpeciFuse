"""Loss functions for SpeciFuse training.

The BarlowTwins, Barlow_Loss, and Barlow_Loss4 classes below are copied from
the original model file without changing their computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import SSIM
from torchvision.transforms.functional import gaussian_blur


class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)

    def forward(self, output, target):
        ssim_loss = 1 - self.ssim(output, target)
        return ssim_loss

    def rgb2gray(self, image):
        b, c, h, w = image.size()
        if c == 1:
            return image
        image_gray = (
            0.299 * image[:, 0, :, :]
            + 0.587 * image[:, 1, :, :]
            + 0.114 * image[:, 2, :, :]
        )
        image_gray = image_gray.unsqueeze(dim=1)
        return image_gray


class L_Intensity_Max_RGB(nn.Module):
    def __init__(self):
        super(L_Intensity_Max_RGB, self).__init__()

    def forward(self, image_visible, image_infrared, image_fused):
        gray_visible = torch.mean(image_visible, dim=1, keepdim=True)
        gray_infrared = torch.mean(image_infrared, dim=1, keepdim=True)

        mask = (gray_infrared > gray_visible).float()
        fused_image = mask * image_infrared + (1 - mask) * image_visible
        loss_intensity = F.l1_loss(fused_image, image_fused)
        return loss_intensity


class IntensityLoss1(nn.Module):
    def __init__(self, kernel_size=11, sigma=1.5):
        super(IntensityLoss1, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma

    def local_entropy(self, x, eps=1e-8):
        b, c, h, w = x.shape

        if c > 1:
            x_gray = 0.299 * x[:, 0:1, :, :] + 0.587 * x[:, 1:2, :, :] + 0.114 * x[:, 2:3, :, :]
        else:
            x_gray = x

        patch_area = self.kernel_size * self.kernel_size

        local_mean = F.avg_pool2d(
            x_gray,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2,
        )

        local_var = (
            F.avg_pool2d(
                x_gray**2,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
            )
            - local_mean**2
        )

        entropy_map = torch.log(local_var + eps)
        entropy_map_smoothed = gaussian_blur(
            entropy_map,
            kernel_size=self.kernel_size,
            sigma=self.sigma,
        )
        return entropy_map_smoothed

    def local_entropy_alternative(self, x, eps=1e-8):
        b, c, h, w = x.shape

        if c > 1:
            x_gray = 0.299 * x[:, 0:1, :, :] + 0.587 * x[:, 1:2, :, :] + 0.114 * x[:, 2:3, :, :]
        else:
            x_gray = x

        patches = F.unfold(
            x_gray,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            stride=1,
        )

        patch_mean = patches.mean(dim=1, keepdim=True)
        patch_std = patches.std(dim=1, keepdim=True)
        entropy = torch.log(patch_std + eps)
        entropy_map = entropy.view(b, 1, h, w)
        entropy_map_smoothed = gaussian_blur(
            entropy_map,
            kernel_size=self.kernel_size,
            sigma=self.sigma,
        )
        return entropy_map_smoothed

    def forward(self, visible_img, infrared_img, fused_img, use_alternative_entropy=False):
        L_vis = F.l1_loss(fused_img, visible_img)

        if use_alternative_entropy:
            en_visible = self.local_entropy_alternative(visible_img)
            en_infrared = self.local_entropy_alternative(infrared_img)
        else:
            en_visible = self.local_entropy(visible_img)
            en_infrared = self.local_entropy(infrared_img)

        if visible_img.shape[1] > 1:
            visible_gray = (
                0.299 * visible_img[:, 0:1, :, :]
                + 0.587 * visible_img[:, 1:2, :, :]
                + 0.114 * visible_img[:, 2:3, :, :]
            )
        else:
            visible_gray = visible_img

        if infrared_img.shape[1] > 1:
            infrared_gray = (
                0.299 * infrared_img[:, 0:1, :, :]
                + 0.587 * infrared_img[:, 1:2, :, :]
                + 0.114 * infrared_img[:, 2:3, :, :]
            )
        else:
            infrared_gray = infrared_img

        intensity_condition = infrared_gray > visible_gray
        entropy_condition = en_infrared > en_visible
        mask = (intensity_condition | entropy_condition).float()

        L_ir = torch.mean(mask * torch.abs(fused_img - infrared_img))
        L_total = L_vis + 2 * L_ir
        return L_total


class GradientMaxLoss(nn.Module):
    def __init__(self):
        super(GradientMaxLoss, self).__init__()
        self.sobel_x = nn.Parameter(
            torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3),
            requires_grad=False,
        ).cuda()
        self.sobel_y = nn.Parameter(
            torch.FloatTensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3),
            requires_grad=False,
        ).cuda()
        self.padding = (1, 1, 1, 1)

    def forward(self, image_A, image_B, image_fuse):
        image_A_gray = self.rgb2gray(image_A)
        image_B_gray = self.rgb2gray(image_B)
        image_fused_gray = self.rgb2gray(image_fuse)
        gradient_A_x, gradient_A_y = self.gradient(image_A_gray)
        gradient_B_x, gradient_B_y = self.gradient(image_B_gray)
        gradient_fuse_x, gradient_fuse_y = self.gradient(image_fused_gray)
        loss = F.l1_loss(gradient_fuse_x, torch.max(gradient_A_x, gradient_B_x)) + F.l1_loss(
            gradient_fuse_y,
            torch.max(gradient_A_y, gradient_B_y),
        )
        return loss

    def gradient(self, image):
        image = F.pad(image, self.padding, mode="replicate")
        gradient_x = F.conv2d(image, self.sobel_x, padding=0)
        gradient_y = F.conv2d(image, self.sobel_y, padding=0)
        return torch.abs(gradient_x), torch.abs(gradient_y)

    def rgb2gray(self, image):
        b, c, h, w = image.size()
        if c == 1:
            return image
        image_gray = (
            0.299 * image[:, 0, :, :]
            + 0.587 * image[:, 1, :, :]
            + 0.114 * image[:, 2, :, :]
        )
        image_gray = image_gray.unsqueeze(dim=1)
        return image_gray


class BarlowTwins(nn.Module):
    def __init__(self):
        super().__init__()
        projector = '8192-8192-8192'
        self.lambd = 0.0051
        sizes = [32] + list(map(int, projector.split('-')))  # 关键修复

        layers = []
        start_dim = 32



        for i in range(len(sizes) - 2):
            layers.append(
                nn.Conv2d(int(start_dim / (pow(2, i))), int(start_dim / (pow(2, i + 1))), kernel_size=3, stride=1,
                          padding=1, bias=False))
            layers.append(nn.BatchNorm2d(int(start_dim / (pow(2, i + 1))), affine=False))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(kernel_size=2))
        layers.append(nn.Conv2d(int(start_dim / 4), int(start_dim / 8), kernel_size=3, stride=1, padding=1, bias=False))

        self.projector = nn.Sequential(*layers)
        self.bn = nn.BatchNorm1d(4096, affine=False)  # 保持4096维度

    def forward(self, y1):
        z1 = self.projector(y1)

        return z1

class Barlow_Loss(nn.Module):
    def __init__(self):
        super(Barlow_Loss, self).__init__()
        self.bn = nn.BatchNorm1d((256 // 4) * (256 // 4), affine=False)

        self.lambd = 0.0051
        self.projector = BarlowTwins().cuda()


    def forward(self, ins1, ins2, ins3):
        inst1 = self.projector(ins1)
        inst2 = self.projector(ins2)
        inst3 = self.projector(ins3)

        con_loss = self.latent_contrast(inst1, inst2, inst3)
        return con_loss

    def off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def cor_mat(self, z1, z2):
        z1 = z1.view(z1.shape[0], -1)
        z2 = z2.view(z2.shape[0], -1)

        c = self.bn(z1).T @ self.bn(z2)
        c.div_(8)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(c).pow_(2).sum()
        return on_diag + self.lambd * off_diag

    def latent_contrast(self, ins1, ins2, ins3):
        
        con_loss = (self.cor_mat(ins1, ins2) +
                    self.cor_mat(ins1, ins3) +
                    self.cor_mat(ins2, ins3))/3
        return con_loss

class Barlow_Loss4(nn.Module):
    def __init__(self):
        super(Barlow_Loss4, self).__init__()
        self.bn = nn.BatchNorm1d((256 // 4) * (256 // 4), affine=False)

        self.lambd = 0.0051
        self.projector = BarlowTwins().cuda()


    def forward(self, ins1, ins2, ins3, ins4):
        inst1 = self.projector(ins1)
        inst2 = self.projector(ins2)
        inst3 = self.projector(ins3)
        inst4 = self.projector(ins4)

        con_loss = self.latent_contrast(inst1, inst2, inst3, inst4)
        return con_loss

    def off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def cor_mat(self, z1, z2):
        z1 = z1.view(z1.shape[0], -1)
        z2 = z2.view(z2.shape[0], -1)

        c = self.bn(z1).T @ self.bn(z2)
        c.div_(8)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(c).pow_(2).sum()
        return on_diag + self.lambd * off_diag

    def latent_contrast(self, ins1, ins2, ins3, ins4):
        
        # symmetric
        con_loss = (self.cor_mat(ins1, ins2) +
                    self.cor_mat(ins1, ins3) +
                    self.cor_mat(ins1, ins4) +
                    self.cor_mat(ins2, ins3)+
                    self.cor_mat(ins2, ins4) +
                    self.cor_mat(ins3, ins4)) / 6
        return con_loss
