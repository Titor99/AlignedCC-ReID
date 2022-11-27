import cv2
import numpy as np
import os
import torch
from utils import model_manager_mine
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

model = model_manager_mine.init_model(name='pcb_seg', num_classes=150)
model = torch.nn.DataParallel(model).cuda()
last_model_wts = torch.load(os.path.join('../logs/prcc_mine', 'checkpoint_best.pth'))
# last_model_wts = torch.load(os.path.join('../imagenet', 'pcb_seg_prcc_hp.pth'))
model.load_state_dict(last_model_wts['state_dict'])
model.eval()
target_layer = [model.module.base.layer4]

# image_path = '/home/wuzhiyue/PycharmProjects/datasets/prcc/rgb/test/C/001/cropped_rgb010.jpg'
image_path = '/home/wuzhiyue/PycharmProjects/datasets/prcc/rgb/train/092/A_cropped_rgb010.jpg'
rgb_img = cv2.imread(image_path, 1)
rgb_img = cv2.resize(rgb_img, (128, 256))
rgb_img = np.float32(rgb_img) / 255


input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

cam = GradCAM(model=model, target_layers=target_layer, use_cuda=False)

target_category = None

grayscale_cam = cam(input_tensor=input_tensor, targets=target_category)

grayscale_cam = grayscale_cam[0]
visualiztion = show_cam_on_image(rgb_img, grayscale_cam)
cv2.imwrite(f'../heatmap.jpg', visualiztion)