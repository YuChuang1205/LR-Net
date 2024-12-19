#!/usr/bin/python3
# coding = gbk
"""
@Author : yuchuang,zhaojinmiao
@Time : 2024/7/4 0:20
@desc: LR-Net
"""
from tqdm import tqdm
import torch.optim as optim
from loss.Edge_loss import edgeSCE_loss
from model.LRNet import LRNet
from components.metric_all_2 import *
from datetime import datetime
import torch
import os
import sys
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from components.utils_all_edge_copy_paste_final_2_img_path import get_loaders, make_dir
from torch.autograd import Variable

# ----------------获取当前运行文件的文件名------------------#
file_path = os.path.abspath(sys.argv[0])
file_name = os.path.basename(file_path)
file_name_without_ext = os.path.splitext(file_name)[0]
# ------------------------------------------------------#

MODEL_NAME = file_name_without_ext

# Hyperparameters etc.
LEARNING_RATE = 5e-4
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_BATCH_SIZE = 10
TEST_BATCH_SIZE = 1
TEST_PATCH_BATCH_SIZE = 48
NUM_EPOCHS = 300
NUM_WORKERS = 5
PATCH_SIZE = 256
PIN_MEMORY = True
LOAD_MODEL = False
root_path = os.path.abspath('.')
TRAIN_IMG_DIR = root_path + "/dataset/Dataset_ICPR/train/img"
TRAIN_MASK_DIR = root_path + "/dataset/Dataset_ICPR/train/mask"
VAL_IMG_DIR = root_path + "/dataset/Dataset_ICPR/val/img"
VAL_MASK_DIR = root_path + "/dataset/Dataset_ICPR/val/mask"
num_images = len(os.listdir(VAL_MASK_DIR))


def PadImg(image, times=32):
    h, w, c = image.shape
    pad_height = (h // times + 1) * times - h
    pad_width = (w // times + 1) * times - w
    image = np.pad(image, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant')
    return image

def test_pred(img, net, batch_size = TEST_PATCH_BATCH_SIZE):
    b, c, h, w = img.shape
    patch_size = PATCH_SIZE
    stride = PATCH_SIZE

    if h > patch_size and w > patch_size:
        img_unfold = F.unfold(img, kernel_size=patch_size, stride=stride)
        img_unfold = img_unfold.reshape(b, c, patch_size, patch_size, -1).permute(0, 4, 1, 2, 3)
        patch_num = img_unfold.size(1)

        preds_list = []
        for i in range(0, patch_num, batch_size):
            end = min(i + batch_size, patch_num)
            batch_patches = img_unfold[:, i:end, :, :, :].reshape(-1, c, patch_size, patch_size)
            batch_patches = Variable(batch_patches.float())
            batch_preds = net.forward(batch_patches)
            preds_list.append(batch_preds)

        # Concatenate all the patch predictions
        preds_unfold = torch.cat(preds_list, dim=0).permute(1, 2, 3, 0)
        preds_unfold = preds_unfold.reshape(b, -1, patch_num)
        preds = F.fold(preds_unfold, kernel_size=patch_size, stride=stride, output_size=(h, w))
    else:
        preds = net.forward(img)

    return preds


def main():
    print(DEVICE)
    def train_fn(loader, model, optimizer, loss_fn, scaler, epoch):
        model.train()
        loop = tqdm(loader)
        iou_metric.reset()
        nIoU_metric.reset()
        train_losses = []
        for batch_idx, (data, targets, edge) in enumerate(loop):
            data = data.to(device=DEVICE).clone().detach()
            targets = targets.unsqueeze(1).to(device=DEVICE).clone().detach()
            edge = edge.to(device=DEVICE).clone().detach()

            with torch.cuda.amp.autocast():
                predictions_no_sigmoid = model(data)
                loss = loss_fn(predictions_no_sigmoid, targets, edge)

            predictions = torch.sigmoid(predictions_no_sigmoid)
            iou_metric.update(predictions, targets)
            nIoU_metric.update(predictions, targets)
            _, IoU = iou_metric.get()
            _, nIoU = nIoU_metric.get()

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            train_losses.append(loss.item())
            scaler.step(optimizer)
            scaler.update()

            loop.set_description(f"train epoch is：{epoch + 1} ")
            loop.set_postfix(loss=loss.item(), IoU=IoU.item(), nIoU=nIoU.item())

        return IoU, nIoU, np.mean(train_losses)


    def val_fn(loader, model, loss_fn, epoch):
        model.eval()
        loop = tqdm(loader)
        iou_metric.reset()
        nIoU_metric.reset()
        FA_PD_metric.reset()
        eval_losses = []
        with torch.no_grad():
            for batch_idx, (x, y, h, w) in enumerate(loop):
                x = x.to(device=DEVICE).clone().detach()
                y = y.unsqueeze(1).to(device=DEVICE).clone().detach()
                preds_no_sigmoid = test_pred(x, model)
                # preds_no_sigmoid = preds_no_sigmoid.cpu().data.numpy()
                preds_no_sigmoid = preds_no_sigmoid[0, :, :h, :w]
                y = y[0, :, :h, :w]
                eval_losses = 0
                preds = torch.sigmoid(preds_no_sigmoid)
                iou_metric.update(preds, y)
                nIoU_metric.update(preds, y)
                FA_PD_metric.update(preds, y)
                _, IoU = iou_metric.get()
                _, nIoU = nIoU_metric.get()
                FA, PD = FA_PD_metric.get(num_images)
                loop.set_description(f"test epoch is：{epoch + 1} ")
                loop.set_postfix(IoU=IoU.item(), nIoU=nIoU.item())
        return IoU, nIoU, np.mean(eval_losses), FA, PD

    train_transform = A.Compose(
        [
            # A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.SomeOf([
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.Transpose(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightness(limit=0.2, p=0.2),
                A.RandomContrast(limit=0.2, p=0.2),
                A.Rotate(limit=45, p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0, rotate_limit=0, p=0.5),
                A.ShiftScaleRotate(shift_limit=0, scale_limit=0.2, rotate_limit=0, p=0.5),
                A.GaussNoise(var_limit=(10.0, 50.0), mean=0, always_apply=False, p=0.2),
                A.NoOp(),
                A.NoOp(),
            ], 3, p=0.5),
            A.Normalize(
                mean=0.2517,
                std=0.0961,
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    val_transforms = A.Compose(
        [
            # A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=0.2517,
                std=0.0961,
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = LRNet().to(DEVICE)
    loss_fn = edgeSCE_loss
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    iou_metric = SigmoidMetric()
    nIoU_metric = SamplewiseSigmoidMetric(1, score_thresh=0.5)
    FA_PD_metric = PD_FA_2(1)
    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        PATCH_SIZE,
        TRAIN_BATCH_SIZE,
        TEST_BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    scaler = torch.cuda.amp.GradScaler()

    best_mIoU = 0
    best_nIoU = 0
    best_PD = 0
    bestmIoUandPD = 0
    bestmIoUandPD_mIoU = 0
    bestmIoUandPD_nIoU = 0
    bestmIoUandPD_FA = 0
    bestmIoUandPD_PD = 0
    bestmIoUandPD_epoch = 0
    best_mIoU_nIoU = 0
    best_mIoU_FA = 0
    best_mIoU_PD = 0
    best_mIoU_epoch = 0
    best_nIoU_mIoU = 0
    best_nIoU_FA = 0
    best_nIoU_PD = 0
    best_nIoU_epoch = 0
    bestPD_mIou = 0
    bestPD_nIou = 0
    bestPD_FA = 0
    bestPD_epoch = 0
    num_epoch = []
    num_train_loss = []
    num_test_loss = []
    num_mioU = []
    num_nioU = []


    save_model_file_path = os.path.join(root_path, 'work_dirs', MODEL_NAME)
    make_dir(save_model_file_path)
    save_file_name = os.path.join(save_model_file_path, MODEL_NAME + '.txt')
    save_bestmiouandPD_file_name = os.path.join(save_model_file_path,
                                                'bestmIoUandPD_checkpoint_' + MODEL_NAME + ".pth.tar")
    save_best_miou_file_name = os.path.join(save_model_file_path,
                                            'best_mIoU_checkpoint_' + MODEL_NAME + ".pth.tar")
    save_best_PD_file_name = os.path.join(save_model_file_path,
                                          'best_PD_checkpoint_' + MODEL_NAME + ".pth.tar")

    save_file = open(save_file_name, 'a')
    start_epoch = 0
    RESUME = False
    if RESUME:
        # torch.cuda.empty_cache()
        path_checkpoint = "./work_dirs/*****/"
        checkpoint = torch.load(path_checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_mIoU = checkpoint['best_mIoU']
        best_nIoU = checkpoint['best_nIoU']
        best_PD = checkpoint['best_PD']
        bestmIoUandPD = checkpoint['bestmIoUandPD']
    # save_file = open('./work_dirs/result_' + MODEL_NAME + '.txt', 'a')
    save_file.write(
        '\n---------------------------------------start--------------------------------------------------\n')
    save_file.write(datetime.now().strftime("%Y-%m-%d, %H:%M:%S\n"))
    for epoch in range(start_epoch, NUM_EPOCHS):
        print(epoch)
        train_mioU, train_nioU, train_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler, epoch)


        mioU, nioU, test_loss, fa, pd = val_fn(val_loader, model, loss_fn, epoch)
        mIoUandPD = 0.5 * mioU + 0.5 * pd
        # save model
        checkpoint = {
            'epoch': epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            'train_loss': train_loss,
            'test_loss': test_loss,
            'best_mIoU': best_mIoU,
            'best_nIoU': best_nIoU,
            'best_PD': best_PD,
            'bestmIoUandPD': bestmIoUandPD,
            'mIoU': mioU,
            'nIoU': nioU,
            'FA': fa,
            'PD': pd,
        }

        if epoch + 1 > 1:
            save_model_file_name = os.path.join(save_model_file_path,
                                                'model_' + MODEL_NAME + '_epoch_' + str(epoch + 1) + '.pth.tar')
            torch.save(checkpoint, save_model_file_name)

        num_epoch.append(epoch + 1)
        num_train_loss.append(train_loss)
        num_test_loss.append(test_loss)
        num_mioU.append(mioU)
        num_nioU.append(nioU)

        if bestmIoUandPD < mIoUandPD:
            bestmIoUandPD = mIoUandPD
            bestmIoUandPD_mIoU = mioU
            bestmIoUandPD_nIoU = nioU
            bestmIoUandPD_FA = fa
            bestmIoUandPD_PD = pd
            bestmIoUandPD_epoch = epoch
            if epoch + 1 > 50:
                torch.save(checkpoint, save_bestmiouandPD_file_name)
        if best_mIoU < mioU:
            best_mIoU = mioU
            best_mIoU_nIoU = nioU
            best_mIoU_FA = fa
            best_mIoU_PD = pd
            best_mIoU_epoch = epoch
            if epoch + 1 > 50:
                torch.save(checkpoint, save_best_miou_file_name)

        if best_PD < pd:
            best_PD = pd
            bestPD_mIou = mioU
            bestPD_nIou = nioU
            bestPD_FA = fa
            bestPD_epoch = epoch
            if epoch + 1 > 50:
                torch.save(checkpoint, save_best_PD_file_name)

        print(f"当前epoch:{epoch + 1}  train_mioU:{round(train_mioU, 4)}  train_nioU:{round(train_nioU, 4)} \n"
              f"当前epoch:{epoch + 1}  mioU:{round(mioU, 4)}  nioU:{round(nioU, 4)}  FA:{round(fa * 1000000, 3)}  PD:{round(pd, 4)} \n"
              f"best_epoch:{best_mIoU_epoch + 1}  best_miou:{round(best_mIoU, 4)}  b_niou:{round(best_mIoU_nIoU, 4)}  FA:{round(best_mIoU_FA * 1000000, 3)}  PD:{round(best_mIoU_PD, 4)}\n"
              f"best_epoch:{bestPD_epoch + 1}  b_miou:{round(bestPD_mIou, 4)}  best_niou:{round(bestPD_nIou, 4)}  FA:{round(bestPD_FA * 1000000, 3)}  PD:{round(best_PD, 4)}\n"
              f"best_epoch:{bestmIoUandPD_epoch + 1}  bestmIoUandPD:{round(bestmIoUandPD, 4)}  bestmIoUandPD_miou:{round(bestmIoUandPD_mIoU, 4)}  bestmIoUandPD_nioU:{round(bestmIoUandPD_nIoU, 4)}  FA:{round(bestmIoUandPD_FA * 1000000, 3)}  PD:{round(bestmIoUandPD_PD, 4)}")

        save_file.write(f"当前epoch:{epoch + 1}  train_mioU:{round(train_mioU, 4)}  train_nioU:{round(train_nioU, 4)} \n")
        save_file.write(
            f"epoch is:{epoch + 1}  mioU:{round(mioU, 4)}  nioU:{round(nioU, 4)}  FA:{round(fa, 3)}  PD:{round(pd, 4)}\n")
        save_file.write(
            f"best_epoch:{best_mIoU_epoch + 1}  best_miou:{round(best_mIoU, 4)}  b_niou:{round(best_mIoU_nIoU, 4)}  FA:{round(best_mIoU_FA * 1000000, 3)}  PD:{round(best_mIoU_PD, 4)}\n")
        save_file.write(
            f"best_epoch:{bestPD_epoch + 1}  b_miou:{round(bestPD_mIou, 4)}  best_niou:{round(bestPD_nIou, 4)}  FA:{round(bestPD_FA * 1000000, 3)}  PD:{round(best_PD, 4)}\n")
        save_file.write(
            f"best_epoch:{bestmIoUandPD_epoch + 1}  bestmIoUandPD:{round(bestmIoUandPD, 4)}  bestmIoUandPD_miou:{round(bestmIoUandPD_mIoU, 4)}  bestmIoUandPD_nioU:{round(bestmIoUandPD_nIoU, 4)}   FA:{round(bestmIoUandPD_FA * 1000000, 3)}  PD:{round(bestmIoUandPD_PD, 4)}\n")

    save_file.write(datetime.now().strftime("%Y-%m-%d, %H:%M:%S\n"))
    save_file.write('\n---------------------------------------end--------------------------------------------------\n')


if __name__ == "__main__":
    main()
