import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from torchvision import datasets
import torchvision.transforms as tvtransforms

import streamlit as st

import matplotlib.pyplot as plt
from PIL import Image


class HardMNIST(datasets.MNIST):
    def __init__(self, root, img_output_shape, noise_level_max, noise_char_max, pos_x, pos_y, digit_out_size, bg_digits,
                 train=True, transform=None, target_transform=None, download=False):
        super(HardMNIST, self).__init__(root, train=train, transform=transform, target_transform=target_transform,
                                        download=download)
        # transformation parameters
        self.img_output_shape = img_output_shape  # new output size (still quadratic)
        self.noise_level_max = noise_level_max  # max. percentage of additive Gaussian noise
        self.noise_char_max = noise_char_max  # max. perentage of salt'n'pepper digit noise
        self.max_rotation = 45  # includes pos and negative directions
        self.max_non_trgts = 10  # maximum number of non-target digits in the background
        self.posx = pos_x
        self.posy = pos_y
        self.digit_out_size = digit_out_size
        self.bg_digits = bg_digits
        # store PIL images of mnist digits
        self.pil_imgs = list()
        for i in range(len(self.data)):
            self.pil_imgs.append(tvtransforms.ToPILImage()(self.data[i]).convert())

    def transform_mnist(self, idx, new_size=28, noise=0.5,
                        sp_noise=0.1, rotate=0, intensity=1.0):
        # digit_img = transforms.ToPILImage()(img).convert()
        digit_img = self.pil_imgs[idx]
        digit_img = digit_img.rotate(rotate, expand=True)
        digit_img = digit_img.resize((new_size, new_size))
        digit_img = torch.Tensor(digit_img.getdata()).float().view(new_size, new_size)

        digit_img *= intensity
        digit_img += 255. * noise * torch.rand(new_size, new_size)
        digit_img[digit_img > 255.] = 255.

        num_pixels = np.int(new_size * new_size * sp_noise)
        pixels = np.random.permutation(new_size * new_size)[:num_pixels]

        pixels2d = np.unravel_index(pixels, (new_size, new_size))
        if sp_noise > 1e-3:
            rnd_img = 255. * noise * torch.rand(new_size, new_size)
            digit_img[pixels2d] = rnd_img[pixels2d]
        return digit_img

    def __getitem__(self, index):
        target = int(self.targets[index])
        non_trgt_inds = np.where(self.targets != target)[0]

        # select some non-target background digits
        non_trgts = self.bg_digits
        non_trgt_inds = np.random.permutation(non_trgt_inds)[:non_trgts]

        # background noise level
        noise_level = self.noise_level_max * np.random.rand()
        new_img = 255. * noise_level * torch.rand(self.img_output_shape, self.img_output_shape)

        for ind in non_trgt_inds:
            # noise_char_level = np.max([0.25, np.random.rand()])
            noise_char_level = 0.
            rotation = 2 * self.max_rotation * np.random.rand() - self.max_rotation + 180
            # scale this mnist digit img to this size
            digit_size = np.max([20, np.int(self.img_output_shape / 4 * np.random.rand())])
            digit_img = self.transform_mnist(ind,
                                             new_size=digit_size,
                                             noise=noise_level,
                                             sp_noise=noise_char_level,
                                             rotate=rotation,
                                             intensity=0.65)
            pos_x = np.int((self.img_output_shape - digit_size) * np.random.rand())
            pos_y = np.int((self.img_output_shape - digit_size) * np.random.rand())
            new_img[pos_y:pos_y + digit_size, pos_x:pos_x + digit_size] = digit_img
            # new_img[pos_y:pos_y + digit_size, pos_x:pos_x + digit_size] /= 2.

        noise_char_level = self.noise_char_max * np.random.rand()
        digit_size = self.digit_out_size
        digit_img = self.transform_mnist(index,
                                         new_size=digit_size,
                                         noise=noise_level,
                                         sp_noise=noise_char_level, rotate=0)

        pos_x = np.min([self.posx, self.img_output_shape - digit_size])
        pos_y = np.min([self.posy, self.img_output_shape - digit_size])
        new_img[pos_y:pos_y + digit_size, pos_x:pos_x + digit_size] = digit_img

        new_img /= 256.
        final_img = Image.fromarray(new_img.numpy(), mode='F').resize((self.img_output_shape, self.img_output_shape))

        if self.transform is not None:
            final_img = self.transform(final_img)
        return final_img, target


class SoftAttnClassifier(nn.Module):

    def __init__(self, img_output_shape, att_scale=10.):
        super(SoftAttnClassifier, self).__init__()
        self.att_scale = att_scale
        self.img_size = img_output_shape
        self.mask_size = np.int(self.img_size / 2.)
        # Feature extractor
        self.fe = nn.Sequential(
            nn.Conv2d(1, 16, 5, 1, 2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        # Upsample layer
        self.upsample = nn.Sequential(
            # upsample part
            nn.Upsample(size=(self.mask_size, self.mask_size), mode='nearest'),
            nn.Conv2d(32, 4, 1, 1),
            nn.BatchNorm2d(4),
        )
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(4 * self.mask_size * self.mask_size, 500, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(500, 10, bias=False),
            nn.LogSoftmax(dim=1),
        )
        # Attention sub-net
        self.fc_attn_size = np.int(self.img_size / 2. / 2.)
        self.foo_1 = torch.arange(0, self.mask_size, device='cpu', dtype=torch.get_default_dtype())[None,
                     :] / np.float(
            self.mask_size)
        self.attn_conv = nn.Conv2d(32, 1, 1, 1)
        self.attn = nn.Sequential(
            nn.Linear(self.fc_attn_size * self.fc_attn_size, 500, bias=True),
            nn.Sigmoid(),
            nn.Linear(500, 4, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        xs = self.fe(x)
        self.mask, self.theta = self.attention(xs)
        xs = self.upsample(xs)
        # (Hadarmad product: feature map * attention mask
        # xy' * mask = diag_x mask diag_y)
        y = xs * self.mask
        self.glimpse = y
        z = y.view(-1, self.mask_size * self.mask_size * 4)
        return self.classifier(z), self.theta[:, 2:]

    def attention(self, x):
        num_samples = x.size()[0]
        foo = self.foo_1.repeat(num_samples, 1)
        z = self.attn_conv(x)
        z = self.attn(z.view(-1, self.fc_attn_size * self.fc_attn_size))

        loc_x = z[:, 0].view(-1, 1).repeat(1, self.mask_size)
        loc_y = z[:, 1].view(-1, 1).repeat(1, self.mask_size)

        sigma_x = z[:, 2].view(-1, 1).repeat(1, self.mask_size)
        sigma_y = z[:, 3].view(-1, 1).repeat(1, self.mask_size)

        x_vec = torch.exp(-(foo - loc_x) * (foo - loc_x) / sigma_x)
        y_vec = torch.exp(-(foo - loc_y) * (foo - loc_y) / sigma_y)

        a_x = x_vec.div(x_vec.sum(dim=1).view(-1, 1).repeat(1, self.mask_size) / self.att_scale)
        a_y = y_vec.div(y_vec.sum(dim=1).view(-1, 1).repeat(1, self.mask_size) / self.att_scale)
        xa = torch.matmul(a_x.view(-1, self.mask_size, 1),
                          a_y.view(-1, 1, self.mask_size)).view(
            -1, 1, self.mask_size, self.mask_size)
        return xa, z


def get_activation(activation, name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


def load_model(img_output_shape):
    model = SoftAttnClassifier(img_output_shape).to('cpu')
    model.load_state_dict(torch.load("soft-attn/weights.pt", map_location=torch.device('cpu')))
    model.eval()
    return model


img_output_shape = 100
img_noise_level = 1

param_choice = st.selectbox('Choose the class label', [*np.arange(10).tolist()])

param_size = st.slider('Digit size', 20, img_output_shape-1, 25, step=1)
max_pos = img_output_shape - param_size
param_posx = st.slider('Horizontal position', 0, max_pos, min([30, max_pos]), step=1)
param_posy = st.slider('Vertical position', 0, max_pos, min([40, max_pos]), step=1)
param_bg_digits = st.slider('Number of background noise digits', 0, 10, 10, step=1)

# param_noise_level = st.slider('Background noise level', 0., 1., 0.5, step=0.01)
# param_noise_char = st.slider('Digit noise level', 0., 1., 0., step=0.01)

param_noise_level = 0.95
param_noise_char = 0.

model = load_model(img_output_shape)

transforms = tvtransforms.Compose([tvtransforms.ToTensor(), tvtransforms.Normalize((0.13,), (0.3,))])
data = HardMNIST('/krabbelkiste/data', img_output_shape, param_noise_level, param_noise_char,
                 pos_x=param_posx, pos_y=param_posy, digit_out_size=param_size, bg_digits=param_bg_digits,
                 train=True, download=True, transform=transforms)

# Visualize feature maps
fig = plt.figure(figsize=(9, 2), dpi=200, facecolor='w', edgecolor='k')
trgts = data.targets
inds = np.where(np.array(trgts) == param_choice)[0]
samples = 1
cnt = 1
for i in range(samples):
    img, target = data.__getitem__(inds[i])

    input = img.view(1, 1, 100, 100)

    outputs, _ = model(input)  # TODO
    _, preds = torch.max(outputs, 1)

    st.write('True class is {0} and predicted class is {1}.'.format(param_choice, preds[0]))

    # act = activation['a'].detach().cpu().squeeze()
    gauss = model.theta.detach().cpu().squeeze()

    mask_raw = model.mask[i, 0, :, :].detach().view(model.mask_size, model.mask_size).cpu().numpy()
    mask = np.array(Image.fromarray(mask_raw).resize((model.img_size, model.img_size)))
    glimpse_comb = model.glimpse[i, :, :, :].sum(dim=0)
    glimpse_raw = glimpse_comb.detach().view(model.mask_size, model.mask_size).cpu().numpy()
    glimpse = np.array(Image.fromarray(glimpse_raw).resize((model.img_size, model.img_size)))

    pos_x = gauss[1] * img_output_shape
    pos_y = gauss[0] * img_output_shape
    sigm_x = 3. * gauss[3] * img_output_shape
    sigm_y = 3. * gauss[2] * img_output_shape

    img = img.view(100, 100).numpy()

    plt.subplot(samples, 4, cnt)
    plt.imshow(img)
    plt.xlim([0, img_output_shape])
    plt.ylim([img_output_shape, 0])
    plt.xticks([])
    plt.yticks([])
    if i == 0:
        plt.title('Input image')

    plt.subplot(samples, 4, cnt+1)
    plt.imshow(mask, cmap='inferno')

    plt.xticks([])
    plt.yticks([])
    plt.xlim([0, img_output_shape])
    plt.ylim([img_output_shape, 0])
    if i == 0:
        plt.title('Attention mask')

    plt.subplot(samples, 4, cnt+2)
    plt.imshow(mask, cmap='Reds')
    plt.imshow(img, cmap='gray', alpha=0.35)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(glimpse, cmap='inferno')
    plt.imshow(img, cmap='gray', alpha=0.25)

    plt.xticks([])
    plt.yticks([])
    if i == 0:
        plt.title('Attention glimpse')

    plt.subplot(samples, 4, cnt+3)
    plt.imshow(img, cmap='Greys_r')
    plt.plot(pos_x, pos_y, 'or', linewidth=4)
    plt.hlines(pos_y - sigm_y, pos_x - sigm_x, pos_x + sigm_x, 'r', linewidth=2)
    plt.hlines(pos_y + sigm_y, pos_x - sigm_x, pos_x + sigm_x, 'r', linewidth=2)
    plt.vlines(pos_x - sigm_x, pos_y - sigm_y, pos_y + sigm_y, 'r', linewidth=2)
    plt.vlines(pos_x + sigm_x, pos_y - sigm_y, pos_y + sigm_y, 'r', linewidth=2)
    plt.xlim([0, img_output_shape])
    plt.ylim([img_output_shape, 0])
    plt.xticks([])
    plt.yticks([])
    if i == 0:
        plt.title('Inferred bounding box')

    cnt += 4

st.pyplot(fig)
