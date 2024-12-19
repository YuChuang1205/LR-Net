#!/usr/bin/python3
# coding = gbk
"""
@Author : yuchuang
@Time : 2024/6/2 16:31
@desc:
"""
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset
from PIL import Image
from model.LRNet import LRNet
from skimage import measure
import torch.nn.functional as F
from torch.autograd import Variable


def read_txt(txt_path):
    with open(txt_path, 'r') as file:
        lines = file.readlines()
    image_out_list = [line.strip()+'.png' for line in lines]
    return image_out_list


def make_dir(path):
    if os.path.exists(path) == False:
        os.makedirs(path)


#################################################
## Adjustable Sensitivity (AS) Strategy. paper: "Infrared Small Target Detection based on Adjustable Sensitivity Strategy and Multi-Scale Fusion"
def target_PD(copy_mask, target_mask):
    copy_contours, _ = cv2.findContours(copy_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    target_contours, _ = cv2.findContours(target_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    overwrite_contours = []
    un_overwrite_contours = []

    target_index_sets = []
    for target_contour in target_contours:
        target_contour_mask = np.zeros(copy_mask.shape, np.uint8)
        cv2.fillPoly(target_contour_mask, [target_contour], (255))
        target_index = np.where(target_contour_mask == 255)
        target_index_XmergeY = set(target_index[0] * 1.0 + target_index[1] * 0.0001)
        target_index_sets.append(target_index_XmergeY)

    for copy_contour in copy_contours:
        copy_contour_mask = np.zeros(copy_mask.shape, np.uint8)
        cv2.fillPoly(copy_contour_mask, [copy_contour], (255))
        copy_index = np.where(copy_contour_mask == 255)
        copy_index_XmergeY = set(copy_index[0] * 1.0 + copy_index[1] * 0.0001)

        overlap_found = False
        for target_index_XmergeY in target_index_sets:
            if not copy_index_XmergeY.isdisjoint(target_index_XmergeY):
                overwrite_contours.append(copy_contour)
                overlap_found = True
                break

        if not overlap_found:
            un_overwrite_contours.append(copy_contour)

    for un_overwrite_c in un_overwrite_contours:
        temp_contour_mask = np.zeros(target_mask.shape, np.uint8)
        cv2.fillPoly(temp_contour_mask, [un_overwrite_c], (255))
        temp_mask = measure.label(temp_contour_mask, connectivity=2)
        coord_image = measure.regionprops(temp_mask)
        # print(coord_image[0].centroid)
        (y, x) = coord_image[0].centroid
        target_mask[int(y), int(x)] = 255

    return target_mask




class SirstDataset(Dataset):
    def __init__(self, image_dir, patch_size, transform=None, mode='None'):
        self.image_dir = image_dir
        self.transform = transform
        self.images = np.sort(os.listdir(image_dir))
        self.mode = mode
        self.patch_size = patch_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        image = np.array(Image.open(img_path).convert("RGB"))

        if (self.mode == 'test'):
            times = 32
            h, w, c = image.shape

            pad_height = (h // times + 1) * times - h
            pad_width = (w // times + 1) * times - w

            image = np.pad(image, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant')
            if self.transform is not None:
                augmentations = self.transform(image=image)
                image = augmentations["image"]
            return image, self.images[index], h, w
        else:
            print("输入的模式错误！！！")



# Hyperparameters etc.
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TEST_BATCH_SIZE = 1
NUM_WORKERS = 4
PIN_MEMORY = True
LOAD_MODEL = False
patch_size = 256   # When the sizes of the dataset samples are small, it can be set to a larger value so that no sliding window cropping is performed during the test in the image inference phase.
TEST_PATCH_BATCH_SIZE = 32
root_path = os.path.abspath('.')
input_path = os.path.join(root_path, 'images')
output_path = os.path.join(root_path, 'mask')
make_dir(output_path)

### Method 1: Read the image name from a txt file.
# TEST_NUM = len(os.listdir(input_path))
# txt_path = os.path.join(root_path,'img_idx','eval.txt') #test.txt
# img_list = read_txt(txt_path)

### Method 2: Directly obtain all image names in a folder.
img_list = os.listdir(input_path)




def test_pred(img, net, batch_size, patch_size):
    b, c, h, w = img.shape
    #print(img.shape)
    patch_size = patch_size
    stride = patch_size

    if h > patch_size and w > patch_size:
        # Unfold the image into patches
        img_unfold = F.unfold(img, kernel_size=patch_size, stride=stride)
        img_unfold = img_unfold.reshape(b, c, patch_size, patch_size, -1).permute(0, 4, 1, 2, 3)
        # print(img_unfold.shape)
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
    test_transforms = A.Compose(
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
    test_ds = SirstDataset(
        image_dir=input_path,
        patch_size=patch_size,
        transform=test_transforms,
        mode='test'
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=TEST_BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False,
    )
    model = LRNet().to(DEVICE)

    model.load_state_dict({k.replace('module.', ''): v for k, v in
                             torch.load("./work_dirs/train_model/best_mIoU_checkpoint_train_model.pth.tar", map_location=DEVICE)[
                                 'state_dict'].items()})
    model.eval()

    temp_num = 0

    for idx, (img,name,h,w) in enumerate(test_loader):
        print(idx)
        img = img.to(device=DEVICE)
        with torch.no_grad():
            image_1 = img
            image_2 = torch.flip(img, [2])
            image_3 = torch.flip(img, [3])

            output_1 = test_pred(image_1, model,batch_size = TEST_PATCH_BATCH_SIZE, patch_size = patch_size)
            output_1 = torch.sigmoid(output_1)
            output_1 = output_1[:, :, :h, :w]
            output_1 = output_1.cpu().data.numpy()


            output_2 = test_pred(image_2, model, batch_size = TEST_PATCH_BATCH_SIZE, patch_size = patch_size)
            output_2 = torch.sigmoid(output_2)
            output_2 = torch.flip(output_2, [2])
            output_2 = output_2[:, :, :h, :w]
            output_2 = output_2.cpu().data.numpy()

            output_3 = test_pred(image_3, model,batch_size = TEST_PATCH_BATCH_SIZE, patch_size = patch_size)
            output_3 = torch.sigmoid(output_3)
            output_3 = torch.flip(output_3, [3])
            output_3 = output_3[:, :, :h, :w]
            output_3 = output_3.cpu().data.numpy()


        output = (output_1 + output_2 + output_3)/3
        for i in range(output.shape[0]):
            print(name[i])
            temp_num = temp_num + 1
            pred = output[i]
            pred = pred[0]
            # pred = np.array(pred, dtype='float32')
            # pred = cv2.resize(pred, (int(w[i]), int(h[i])))
            pred_target = np.where(pred > 0.5, 255, 0)  #"0.5" can be adjusted according to different datasets
            pred_target = np.array(pred_target, dtype='uint8')

            pred_copy_0 = np.where(pred > 0.13, 255, 0)    # "0.13" can be adjusted according to different datasets
            pred_copy_0 = np.array(pred_copy_0, dtype='uint8')


            # pred_1 = output_1[i]
            # pred_1 = pred_1[0]
            # pred_1 = np.array(pred_1, dtype='float32')
            # pred_1 = cv2.resize(pred_1, (int(w[i]), int(h[i])))
            # pred_copy_1 = np.where(pred_1 > 0.1, 255, 0)
            # pred_copy_1 = np.array(pred_copy_1, dtype='uint8')
            #
            # pred_2 = output_2[i]
            # pred_2 = pred_2[0]
            # pred_2 = np.array(pred_2, dtype='float32')
            # pred_2 = cv2.resize(pred_2, (int(w[i]), int(h[i])))
            # pred_copy_2 = np.where(pred_2 > 0.1, 255, 0)
            # pred_copy_2 = np.array(pred_copy_2, dtype='uint8')
            #
            # pred_3 = output_3[i]
            # pred_3 = pred_3[0]
            # pred_3 = np.array(pred_3, dtype='float32')
            # pred_3 = cv2.resize(pred_3, (int(w[i]), int(h[i])))
            # pred_copy_3 = np.where(pred_3 > 0.1, 255, 0)
            # pred_copy_3 = np.array(pred_copy_3, dtype='uint8')


            pred_out_0 = target_PD(pred_copy_0, pred_target)
            # pred_out_1 = target_PD(pred_copy_1, pred_out_0)
            # pred_out_2 = target_PD(pred_copy_2, pred_out_1)
            # pred_out_3 = target_PD(pred_copy_3, pred_out_2)

            cv2.imwrite(os.path.join(output_path, name[i]), pred_out_0)

if __name__ == "__main__":
    main()
