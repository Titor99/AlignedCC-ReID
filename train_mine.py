
import argparse
import datetime
import os
import shutil
import torch
import copy
import numpy as np
from tensorboardX import SummaryWriter

from utils import data_manager_mine
from utils.transforms import build_transforms_base
from utils.data_manager_mine import ImageDatasetHpMask
from utils.samplers import RandomIdentitySamplerGcn
from utils.collate_batch import train_collate_gcn_mask, val_collate_gcn_mask

from utils import model_manager_mine
from utils.losses_mine import make_loss
from utils.inference import inference_prcc_global, inference_prcc_visual_rank, inference_prcc_visual_rank_easy
from utils.optimizer import make_optimizer_with_triplet, WarmupMultiStepLR
from torch.utils.data import DataLoader



def main(cfg):


    # loading dataset
    print('loading dataset')

    dataset = data_manager_mine.init_dataset(name=cfg.dataset,root=cfg.data_dir)
    num_classes = dataset.train_data_ids
    num_test = len(dataset.query_data)

    train_transforms = build_transforms_base(cfg, is_train=True)
    test_transforms = build_transforms_base(cfg, is_train=False)

    train_loader = DataLoader(
        ImageDatasetHpMask(dataset.train_data, cfg.height, cfg.width, train_transforms),
        batch_size=cfg.batch_size, sampler=RandomIdentitySamplerGcn(dataset.train_data, cfg.batch_size, cfg.img_per_id),
        num_workers=8, collate_fn=train_collate_gcn_mask)

    test_loader = DataLoader(
        ImageDatasetHpMask(dataset.test_data, cfg.height, cfg.width, test_transforms),batch_size=1, shuffle=False,
        num_workers=8, collate_fn=val_collate_gcn_mask)

    # loading model
    print('loading model')

    model = model_manager_mine.init_model(name=cfg.model, num_classes=num_classes)
    model = torch.nn.DataParallel(model).cuda()

    loss = make_loss(cfg, num_classes)

    trainer = make_optimizer_with_triplet(cfg, model)
    scheduler = WarmupMultiStepLR(trainer, milestones=[40, 80], gamma=0.1, warmup_factor=1.0 / 3, warmup_iters=500,
                                  warmup_method="linear", last_epoch=-1,)

    working_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir_name = str('logs/' + cfg.dataset + '_mine')
    log_dir = os.path.join(working_dir, log_dir_name)

    if cfg.test:
        # Test
        model_wts = torch.load(os.path.join(log_dir, 'checkpoint_best.pth'))
        # model_wts = torch.load('imagenet/pcb_seg_prcc_hp.pth')
        model.load_state_dict(model_wts['state_dict'])

        mAP, cmc1 = inference_prcc_global(model, test_loader, num_test)
        start_time = datetime.datetime.now()
        start_time = '%4d:%d:%d-%2d:%2d:%2d' % (
        start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second)
        line = '{} - Test: cmc1: {:.1%}, mAP: {:.1%}\n'.format(start_time, cmc1, mAP)
        print(line)

    if cfg.visualize:
        # Visualize
        model_wts = torch.load(os.path.join(log_dir, 'checkpoint_best.pth'))
        # model_wts = torch.load('imagenet/pcb_seg_prcc_hp.pth')
        model.load_state_dict(model_wts['state_dict'])

        home = os.path.join('logs', 'visualize', os.path.basename(log_dir))
        if cfg.easy:
            inference_prcc_visual_rank_easy(model, test_loader, num_test, home=home, show_rank=10, use_flip=True)
        else:
            inference_prcc_visual_rank(model, test_loader, num_test, home=home, show_rank=10, use_flip=True)

    if cfg.test == False and cfg.visualize == False:
        # training
        print('start training')
        train(model, train_loader, test_loader, loss, trainer, scheduler, cfg.epochs, num_query=num_test, log_dir=log_dir)

    print('finished')

def train(model, train_loader, test_loader, loss_fn, optimizer, scheduler, num_epochs, num_query, log_dir):

    writer = SummaryWriter(log_dir=log_dir)
    acc_best = 0.0
    last_acc_val = acc_best

    use_cuda = torch.cuda.is_available()

    for epoch in range(num_epochs):
        model.train()

        for ii, (img, target, path, mask) in enumerate(
                train_loader):  # [64, 3, 256, 128],  [64,],  [64,], [64, 6, 256,b   128]
            img = img.cuda() if use_cuda else img  # [64, 3, 256, 128]
            target = target.cuda() if use_cuda else target  # [64,]
            b, c, h, w = img.shape

            mask = mask.cuda() if use_cuda else mask  # [64, 6, 256, 128]
            mask_i = mask.unsqueeze(dim=1)
            mask_i = mask_i.expand_as(img)
            img_a = copy.deepcopy(img)  # [40, 3, 256, 128]

            index = np.random.permutation(b)
            img_r = img[index]  # [64, 3, 256, 128]
            msk_r = mask_i[index]  # [64, 6, 256, 128]
            img_a[mask_i == 5] = img_r[msk_r == 5]

            index = np.random.permutation(b)
            img_r = img[index]  # [64, 3, 256, 128]
            msk_r = mask_i[index]  # [64, 6, 256, 128]
            img_a[mask_i == 6] = img_r[msk_r == 6]

            index = np.random.permutation(b)
            img_r = img[index]  # [64, 3, 256, 128]
            msk_r = mask_i[index]  # [64, 6, 256, 128]
            img_a[mask_i == 7] = img_r[msk_r == 7]

            index = np.random.permutation(b)
            img_r = img[index]  # [64, 3, 256, 128]
            msk_r = mask_i[index]  # [64, 6, 256, 128]
            img_a[mask_i == 9] = img_r[msk_r == 9]

            index = np.random.permutation(b)
            img_r = img[index]  # [64, 3, 256, 128]
            msk_r = mask_i[index]  # [64, 6, 256, 128]
            img_a[mask_i == 10] = img_r[msk_r == 10]

            index = np.random.permutation(b)
            img_r = img[index]  # [64, 3, 256, 128]
            msk_r = mask_i[index]  # [64, 6, 256, 128]
            img_a[mask_i == 12] = img_r[msk_r == 12]
            '''
            img_a[mask_i == 0] = img_r[msk_r == 0]  # background
            img_a[mask_i == 1] = img_r[msk_r == 1]  # hat
            img_a[mask_i == 2] = img_r[msk_r == 2]  # hair
            img_a[mask_i == 3] = img_r[msk_r == 3]  # glove
            img_a[mask_i == 4] = img_r[msk_r == 4]  # sunglasses
            img_a[mask_i == 5] = img_r[msk_r == 5]  # upper cloth
            img_a[mask_i == 6] = img_r[msk_r == 6]  # dress
            img_a[mask_i == 7] = img_r[msk_r == 7]  # coat
            img_a[mask_i == 8] = img_r[msk_r == 8]  # sock
            img_a[mask_i == 9] = img_r[msk_r == 9]  # pants
            img_a[mask_i == 10] = img_r[msk_r == 10]  # jumpsuit
            img_a[mask_i == 11] = img_r[msk_r == 11]  # scarf
            img_a[mask_i == 12] = img_r[msk_r == 12]  # skirt
            img_a[mask_i == 13] = img_r[msk_r == 13]  # face
            img_a[mask_i == 14] = img_r[msk_r == 14]  # left leg
            img_a[mask_i == 15] = img_r[msk_r == 15]  # right leg
            img_a[mask_i == 16] = img_r[msk_r == 16]  # left arm
            img_a[mask_i == 17] = img_r[msk_r == 17]  # right arm
            img_a[mask_i == 18] = img_r[msk_r == 18]  # left shoe
            img_a[mask_i == 19] = img_r[msk_r == 19]  # right shoe

            # img_a[mask_i == 0] = 0
            '''

            img_c = torch.cat([img, img_a], dim=0)  # [80, 3, 256, 128]
            target_c = torch.cat([target, target], dim=0)
            score, feat = model(img_c)  # [64, 150], [64, 2018]

            loss = loss_fn(score, feat, target_c)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # compute acc
        acc = (score.max(1)[1] == target_c).float().mean()
        loss = float(loss)
        acc = float(acc)

        start_time = datetime.datetime.now()
        start_time = '%d/%d-%2d:%2d' % (start_time.month, start_time.day, start_time.hour, start_time.minute)
        print('{} - epoch: {}  Loss: {:.04f}  Acc: {:.1%}  '.format(start_time, epoch, loss, acc))

        if epoch % 5 == 0:
            # test & save model
            mAP, cmc1 = inference_prcc_global(model, test_loader, num_query)
            start_time = datetime.datetime.now()
            start_time = '%d/%d-%2d:%2d' % (start_time.month, start_time.day, start_time.hour, start_time.minute)
            line = '{} - cmc1: {:.1%} mAP: {:.1%}\n'.format(start_time, cmc1, mAP)
            print(line)
            f = open(os.path.join(log_dir, 'logs.txt'), 'a')
            f.write(line)
            f.close()

            acc_test = 0.5 * (cmc1 + mAP)
            is_best = acc_test >= last_acc_val
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_acc': acc_test,
            }, is_best, fpath=log_dir)
            if is_best:
                last_acc_val = acc_test

            writer.add_scalar('train_loss', float(loss), epoch + 1)
            writer.add_scalar('test_rank1', float(cmc1), epoch + 1)
            writer.add_scalar('test_mAP', float(mAP), epoch + 1)

        scheduler.step()

    # Test
    last_model_wts = torch.load(os.path.join(log_dir, 'checkpoint_best.pth'))
    model.load_state_dict(last_model_wts['state_dict'])
    mAP, cmc1 = inference_prcc_global(model, test_loader, num_query)

    start_time = datetime.datetime.now()
    start_time = '%4d:%d:%d-%2d:%2d:%2d' % (start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second)
    line = '{} - Final: cmc1: {:.1%}, mAP: {:.1%}\n'.format(start_time, cmc1, mAP)
    print(line)
    f = open(os.path.join(log_dir, 'logs.txt'), 'a')
    f.write(line)
    f.close()


def save_checkpoint(state, is_best, fpath):
    if not os.path.exists(fpath):
        os.makedirs(fpath)

    fpath = os.path.join(fpath, 'checkpoint.pth')
    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, os.path.join(os.path.dirname(fpath), 'checkpoint_best.pth'))
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Train")

    parser.add_argument('--dataset', type=str, default='prcc')
    parser.add_argument('--model', type=str, default='pcb_seg')
    parser.add_argument('--data_dir', type=str, default='../datasets/prcc')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--img_per_id', type=int, default=4)
    parser.add_argument('--test', action='store_true', help='test')
    parser.add_argument('--visualize', action='store_true', help='visualize')
    parser.add_argument('--easy', action='store_true', help='easy visualize')

    parser.add_argument('--lr', type=float, default=0.00035)
    parser.add_argument('--bias_lr_factor', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--weight_decay_bias', type=float, default=5e-4)
    parser.add_argument('--optimizer_name', type=str, default="SGD", help="SGD, Adam")
    parser.add_argument('--momentum', type=float, default=0.9)

    cfg = parser.parse_args()

    main(cfg)
