# Enable import from parent package
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.tensorboard import SummaryWriter
import numpy as np
import dataio
import utils
import training
import loss_functions
import modules
from torch.utils.data import DataLoader
import configargparse
import torch
from functools import partial

import math


torch.backends.cudnn.benchmark = True
torch.set_num_threads(4)

p = configargparse.ArgumentParser()

# config file, output directories
p.add('-c', '--config', required=False, is_config_file=True,
      help='Path to config file.')
p.add_argument('--logging_root', type=str, default='../logs', help='root for logging')
p.add_argument('--experiment_name', type=str, default='train_img',
               help='path to directory where checkpoints & tensorboard events will be saved.')

# general training options
p.add_argument('--model', default='mfn', choices=['mfn', 'mlp'],
               help='use MFN or standard MLP')
p.add_argument('--batch_size', type=int, default=1)
p.add_argument('--hidden_features', type=int, default=32)
p.add_argument('--hidden_layers', type=int, default=4)
p.add_argument('--res', type=int, default=256,
               help='resolution of image to fit, also used to set the network-equivalent sample rate'
               + ' i.e., the maximum network bandwidth in cycles per unit interval is half this value')
p.add_argument('--lr', type=float, default=5e-4, help='learning rate')
p.add_argument('--num_steps', type=int, default=5001,
               help='number of training steps')
p.add_argument('--gpu', type=int, default=0,
               help='gpu id to use for training')

# mfn options
p.add_argument('--multiscale', action='store_true', default=False,
               help='use multiscale')
p.add_argument('--use_resized', action='store_true', default=False,
               help='use multiscale')

# mlp options
p.add_argument('--activation', type=str, default='sine',
               choices=['sine', 'relu', 'requ', 'gelu', 'selu', 'softplus', 'tanh', 'swish'],
               help='activation to use (for model mlp only)')
p.add_argument('--ipe', action='store_true', default=False,
               help='use integrated positional encoding')
p.add_argument('--w0', type=float, default=10)
p.add_argument('--pe_scale', type=float, default=3, help='positional encoding scale')
p.add_argument('--no_pe', action='store_true', default=False,
               help='override to have no positional encoding for relu mlp')

# data processing and i/o
p.add_argument('--centered', action='store_true', default=False,
               help='centere input coordinates as -1 to 1')
p.add_argument('--img_fn', type=str, default='/ubc/cs/research/kmyi/zwu/home/codes/bacon/data/images/tokyo.jpg',
               help='path to specific png filename')
p.add_argument('--grayscale', action='store_true', default=False,
               help='if grayscale image')

# summary, logging options
p.add_argument('--steps_til_ckpt', type=int, default=100,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=100,
               help='Time interval in seconds until tensorboard summary is saved.')

opt = p.parse_args()

if opt.experiment_name is None and opt.render_model is None:
    p.error('--experiment_name is required.')

# os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)


def main():

    print('--- Run Configuration ---')
    for k, v in vars(opt).items():
        print(k, v)

    train()


def train():
    device = torch.device("cuda")

    # set up logging dir
    opt.root_path = os.path.join(opt.logging_root, opt.experiment_name)
    utils.cond_mkdir(opt.root_path)

    # get datasets
    trn_dataset, val_dataset, dataloader = init_dataloader(opt)

    # set up coordinate network
    model = init_model(opt)

    # loss and tensorboard logging functions
    loss_fn, summary_fn = init_loss(opt, trn_dataset, val_dataset)

    # back up config file
    save_params(opt, model)

    # start training
    training.train(model=model, train_dataloader=dataloader,
                   steps=opt.num_steps, lr=opt.lr,
                   steps_til_summary=opt.steps_til_summary,
                   steps_til_checkpoint=opt.steps_til_ckpt,
                   model_dir=opt.root_path, loss_fn=loss_fn, summary_fn=summary_fn)

    ### Output the final reconstruction image
    path = f"{opt.root_path}/result.jpg"
    print(f"Writing '{path}'... ", end="")

    # resolution = image.data.shape[0:2]
    # img_shape = image.data.shape
    # n_pixels = resolution[0] * resolution[1]

    resolution = val_dataset.img_chw.shape[1:]
    img_shape = resolution + tuple([3])

    half_dx = 0.5 / resolution[0]
    half_dy = 0.5 / resolution[1]
    xs = torch.linspace(half_dx, 1 - half_dx, resolution[0], device=device)
    ys = torch.linspace(half_dy, 1 - half_dy, resolution[1], device=device)
    xv, yv = torch.meshgrid([xs, ys], indexing="ij")

    xy = torch.stack((xv.flatten(), yv.flatten())).t()
    xy = xy - 0.5

    xy_max_num = math.ceil(xy.shape[0] / 1024.0)
    padding_delta = xy_max_num * 1024 - xy.shape[0]
    zeros_padding = torch.zeros((padding_delta, 2)).cuda()
    xy_padded = torch.cat([xy, zeros_padding], dim=0)

    with torch.no_grad():
        pred_img = process_batch_in_chunks(xy_padded, model, max_chunk_size=2 ** 18)
        pred_img = pred_img[:xy.shape[0], :].reshape(img_shape).float().clamp(0.0, 1.0).numpy()
        # Record psnr and ssim
        dataio.save_img(pred_img, path)
    print("done.")


def process_batch_in_chunks(in_ccords, model, max_chunk_size=1024):
    chunk_outs = []
    coord_chunks = torch.split(in_ccords, max_chunk_size)

    for chunk_batched_in in coord_chunks:
        inputs = dict({'coords': chunk_batched_in})
        tmp_img = model(inputs)['model_out']['output'][-1]
        chunk_outs.append(tmp_img.detach().cpu())

    batched_out = torch.cat(chunk_outs, dim=0)
    return batched_out


def init_dataloader(opt):
    ''' load image datasets, dataloader '''

    # if opt.img_fn == '../data/lighthouse.png':
    #     url = 'http://www.cs.albany.edu/~xypan/research/img/Kodak/kodim19.png'
    # else:
    #     url = None

    url = None

    # init datasets
    trn_dataset = dataio.ImageFile(opt.img_fn, grayscale=opt.grayscale, resolution=(opt.res, opt.res), url=url)

    val_dataset = dataio.ImageFile(opt.img_fn, grayscale=opt.grayscale, resolution=(2*opt.res, 2*opt.res), url=url)

    trn_dataset = dataio.ImageWrapper(trn_dataset, centered=opt.centered,
                                      include_end=False,
                                      multiscale=opt.use_resized,
                                      stages=3)

    val_dataset = dataio.ImageWrapper(val_dataset, centered=opt.centered,
                                      include_end=False,
                                      multiscale=opt.use_resized,
                                      stages=3)

    dataloader = DataLoader(trn_dataset, shuffle=True, batch_size=opt.batch_size,
                            pin_memory=True, num_workers=0)

    return trn_dataset, val_dataset, dataloader


def init_model(opt):

    if opt.grayscale:
        out_features = 1
    else:
        out_features = 3

    if opt.model == 'mlp':

        if opt.multiscale:
            m = modules.MultiscaleCoordinateNet
        else:
            m = modules.CoordinateNet

        model = m(nl=opt.activation,
                  in_features=2,
                  out_features=out_features,
                  hidden_features=opt.hidden_features,
                  num_hidden_layers=opt.hidden_layers,
                  w0=opt.w0,
                  pe_scale=opt.pe_scale,
                  no_pe=opt.no_pe,
                  integrated_pe=opt.ipe)

    elif opt.model == 'mfn':

        if opt.multiscale:
            m = modules.MultiscaleBACON
        else:
            m = modules.BACON

        input_scales = [1/8, 1/8, 1/4, 1/4, 1/4]
        output_layers = [1, 2, 4]

        model = m(2, opt.hidden_features, out_size=out_features,
                  hidden_layers=opt.hidden_layers,
                  bias=True,
                  frequency=(opt.res, opt.res),
                  quantization_interval=2*np.pi,
                  input_scales=input_scales,
                  output_layers=output_layers,
                  reuse_filters=False)

    else:
        raise ValueError('model must be mlp or mfn')

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Num. Parameters: {params}')
    model.cuda()

    return model


def init_loss(opt, trn_dataset, val_dataset):
    ''' define loss, summary functions given expmt configs '''

    # initialize the loss
    if opt.multiscale:
        loss_fn = partial(loss_functions.multiscale_image_mse, use_resized=opt.use_resized)
        summary_fn = partial(utils.write_multiscale_image_summary, (opt.res, opt.res),
                             trn_dataset, use_resized=opt.use_resized, val_dataset=val_dataset)
    else:
        loss_fn = loss_functions.image_mse
        summary_fn = partial(utils.write_image_summary, (opt.res, opt.res), trn_dataset,
                             val_dataset=val_dataset)

    return loss_fn, summary_fn


def save_params(opt, model):

    # Save command-line parameters log directory.
    p.write_config_file(opt, [os.path.join(opt.root_path, 'config.ini')])
    with open(os.path.join(opt.root_path, "params.txt"), "w") as out_file:
        out_file.write('\n'.join(["%s: %s" % (key, value) for key, value in vars(opt).items()]))

    # Save text summary of model into log directory.
    with open(os.path.join(opt.root_path, "model.txt"), "w") as out_file:
        out_file.write(str(model))


if __name__ == '__main__':
    main()
