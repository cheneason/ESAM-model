import argparse
import os
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import datasets
import models
import utils
from statistics import mean
import torch
import numpy as np

def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        shuffle=False, num_workers=8, pin_memory=True)
    return loader


def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    return train_loader, val_loader


def eval_psnr(loader, model, eval_type=None):
    model.eval()
    if eval_type == 'f1':
        metric_fn = utils.calc_f1
        metric1, metric2, metric3, metric4 = 'f1', 'auc', 'none', 'none'
    elif eval_type == 'fmeasure':
        metric_fn = utils.calc_fmeasure
        metric1, metric2, metric3, metric4 = 'f_mea', 'mae', 'none', 'none'
    elif eval_type == 'ber':
        metric_fn = utils.calc_ber
        metric1, metric2, metric3, metric4 = 'shadow', 'non_shadow', 'ber', 'none'
    elif eval_type == 'cod':
        metric_fn = utils.calc_cod
        metric1, metric2, metric3, metric4 = 'sm', 'em', 'wfm', 'mae'
    pbar = tqdm(total=len(loader), leave=False, desc='val')

    result1_list = []
    result2_list = []
    result3_list = []
    result4_list = []
    result5_list = []
    result6_list = []
    result7_list = []
    result8_list = []
    result9_list = []
    result10_list = []
    result11_list = []
    result12_list = []

    with torch.no_grad():
        for batch in loader:
            for k, v in batch.items():
                batch[k] = v.cuda()

            inp = batch['inp']
            batch_gt = batch['gt']
            bgt = batch['bgt']

            pred,predb,fine = model.infer(inp)
            pred = torch.sigmoid(pred)
            predb = torch.sigmoid(predb)
            pred_fine = torch.sigmoid(fine)

            result1, result2, result3, result4 = metric_fn(pred, batch_gt)
            result5, result6, result7, result8 = metric_fn(predb, bgt)
            result9, result10, result11, result12 = metric_fn(pred_fine, batch_gt)
            result1_list.append(result1)
            result2_list.append(result2)
            result3_list.append(result3)
            result4_list.append(result4)
            result5_list.append(result5)
            result6_list.append(result6)
            result7_list.append(result7)
            result8_list.append(result8)
            result9_list.append(result9)
            result10_list.append(result10)
            result11_list.append(result11)
            result12_list.append(result12)

            if pbar is not None:
                pbar.update(1)
        if pbar is not None:
            pbar.close()

    result1_mean = np.mean(result1_list)
    result2_mean = np.mean(result2_list)
    result3_mean = np.mean(result3_list)
    result4_mean = np.mean(result4_list)
    result5_mean = np.mean(result5_list)
    result6_mean = np.mean(result6_list)
    result7_mean = np.mean(result7_list)
    result8_mean = np.mean(result8_list)
    result9_mean = np.mean(result9_list)
    result10_mean = np.mean(result10_list)
    result11_mean = np.mean(result11_list)
    result12_mean = np.mean(result12_list)
    return result1_mean, result2_mean, result3_mean, result4_mean,result5_mean, result6_mean, result7_mean, result8_mean,result9_mean, result10_mean, result11_mean, result12_mean,metric1, metric2, metric3, metric4


def prepare_training():
    if config.get('resume') is not None:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = config.get('resume') + 1
    else:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = 1
    max_epoch = config.get('epoch_max')
    lr_scheduler = CosineAnnealingLR(optimizer, max_epoch, eta_min=config.get('lr_min'))
    return model, optimizer, epoch_start, lr_scheduler

def train(train_loader, model, epoch):
    model.train()
    pbar = tqdm(total=len(train_loader), desc=f"train:epoch{epoch}/{config['epoch_max']}")

    loss_list = []
    seg_losslist = []
    edge_losslist = []
    fusion_losslist = []
    for batch in train_loader:
        for k, v in batch.items():
            batch[k] = v.to(device)
        inp = batch['inp']
        gt = batch['gt']
        bgt = batch['bgt']
        model.set_input(inp, gt, bgt)
        model.optimize_parameters()

        batch_loss = model.loss_G.cpu().detach().numpy()
        seg_loss = model.seg_loss.cpu().detach().numpy()
        edge_loss = model.edge_loss.cpu().detach().numpy()
        fusion_loss = model.fusion_loss.cpu().detach().numpy()

        loss_list.append(batch_loss)
        seg_losslist.append(seg_loss)
        edge_losslist.append(edge_loss)
        fusion_losslist.append(fusion_loss)

        if pbar is not None:
            pbar.update(1)
    if pbar is not None:
        pbar.close()

    loss = [i.item() for i in loss_list]
    loss_seg =[j.item() for j in seg_losslist]
    loss_edge = [j.item() for j in edge_losslist]
    loss_fusion = [j.item() for j in fusion_losslist]
    print(mean(loss_seg), mean(loss_edge), mean(loss_fusion))
    return mean(loss)


def main(config_, save_path, args):
    global config, log, writer, log_info
    config = config_
    log, writer = utils.set_save_path(save_path, remove=False)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    train_loader, val_loader = make_data_loaders()
    if config.get('data_norm') is None:
        config['data_norm'] = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

    model, optimizer, epoch_start, lr_scheduler = prepare_training()
    model.optimizer = optimizer
    lr_scheduler = CosineAnnealingLR(model.optimizer, config['epoch_max'], eta_min=config.get('lr_min'))
    model = model.cuda()

    sam_checkpoint = torch.load(config['sam_checkpoint'])
    model.load_state_dict(sam_checkpoint, strict=False)
    for name, para in model.named_parameters():
        if "image_encoder" in name and "prompt_generator" not in name:
            para.requires_grad_(False)

    model_total_params = sum(p.numel() for p in model.parameters())
    model_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('model_grad_params:' + str(model_grad_params), '\nmodel_total_params:' + str(model_total_params))

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    max_val_v = -1e18 if config['eval_type'] != 'ber' else 1e8
    timer = utils.Timer()
    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()
        train_loss_G = train(train_loader, model, epoch)
        lr_scheduler.step()

        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        log_info.append('train G: loss={:.4f}'.format(train_loss_G))
        writer.add_scalars('loss', {'train G': train_loss_G}, epoch)

        model_spec = config['model']
        model_spec['sd'] = model.state_dict()
        optimizer_spec = config['optimizer']
        optimizer_spec['sd'] = optimizer.state_dict()

        save(config, model, save_path, 'last')

        if (epoch_val is not None) and (epoch % epoch_val == 0):
            result1, result2, result3, result4,result5, result6, result7, result8,result9, result10, result11, result12, metric1, metric2, metric3, metric4 = eval_psnr(val_loader, model,
                eval_type=config.get('eval_type'))

            log_info.append('val: {}={:.4f}'.format(metric1, result1))
            writer.add_scalars(metric1, {'val': result1}, epoch)
            log_info.append('val: {}={:.4f}'.format(metric2, result2))
            writer.add_scalars(metric2, {'val': result2}, epoch)
            log_info.append('val: {}={:.4f}'.format(metric3, result3))
            writer.add_scalars(metric3, {'val': result3}, epoch)
            log_info.append('val: {}={:.4f}'.format(metric4, result4))
            writer.add_scalars(metric4, {'val': result4}, epoch)
            log_info.append('val: {}={:.4f}'.format(metric1, result5))
            writer.add_scalars(metric1, {'val': result5}, epoch)
            log_info.append('val: {}={:.4f}'.format(metric2, result6))
            writer.add_scalars(metric2, {'val': result6}, epoch)
            log_info.append('val: {}={:.4f}'.format(metric3, result7))
            writer.add_scalars(metric3, {'val': result7}, epoch)
            log_info.append('val: {}={:.4f}'.format(metric4, result8))
            writer.add_scalars(metric4, {'val': result8}, epoch)
            log_info.append('val: {}={:.4f}'.format(metric1, result9))
            writer.add_scalars(metric1, {'val': result9}, epoch)
            log_info.append('val: {}={:.4f}'.format(metric2, result10))
            writer.add_scalars(metric2, {'val': result10}, epoch)
            log_info.append('val: {}={:.4f}'.format(metric3, result11))
            writer.add_scalars(metric3, {'val': result11}, epoch)
            log_info.append('val: {}={:.4f}'.format(metric4, result12))
            writer.add_scalars(metric4, {'val': result12}, epoch)

            if result11 > max_val_v:
                max_val_v = result11
                save(config, model, save_path, 'best')
            t = timer.t()
            prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
            t_epoch = utils.time_text(t - t_epoch_start)
            t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
            log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))
            log(', '.join(log_info))
            writer.flush()


def save(config, model, save_path, name):
    if config['model']['name'] == 'segformer' or config['model']['name'] == 'setr':
        if config['model']['args']['encoder_mode']['name'] == 'evp':
            prompt_generator = model.encoder.backbone.prompt_generator.state_dict()
            decode_head = model.encoder.decode_head.state_dict()
            torch.save({"prompt": prompt_generator, "decode_head": decode_head},
                       os.path.join(save_path, f"prompt_epoch_{name}.pth"))
        else:
            torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{name}.pth"))
    else:
        torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{name}.pth"))


if __name__ == '__main__':

    train_on_gpu = torch.cuda.is_available()
    if not train_on_gpu:
        print('CUDA is not available. Training on CPU')
    else:
        print('CUDA is available. Training on GPU')
    device = torch.device("cuda:0" if train_on_gpu else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="configs/cod-sam-vit-l.yaml")
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument("--local_rank", type=int, default=-1, help="")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

        print('config loaded.')

    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('./save', save_name)
    main(config, save_path, args=args)

