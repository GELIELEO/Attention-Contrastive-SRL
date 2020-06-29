import torch
import torchvision.transforms as transforms
import cv2
import numpy as np





def augment(img=None, out_size=64):
    # input: C x W x H, np.array
    # output: c x w x h, torch.tensor
    if img is None:
        img = cv2.imread('./utils/test_img.png')
        print('img_aug shape', img.shape)
    else:
        # print('img_aug shape', img.shape)
        img = np.transpose(img, (1, 2, 0)) # to be W x H x C

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    augmentation = [
            transforms.ToPILImage(),
            # transforms.RandomResizedCrop(out_size, scale=(0.2, 1.)),
            # transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomApply([
            #     transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            #     ], p=0.8),
            transforms.ToTensor(),
            normalize
           ]

    transformer = transforms.Compose(augmentation)

    img_T = transformer(img)
    # print('img_T', img_T.shape)

    # cv2.imshow('img', img)
    # cv2.imshow('img_T', img_T.permute(1,2,0).numpy())
    # cv2.waitKey(0)

    return img_T



if __name__ == '__main__':
    augment()



