import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import model
import numpy as np
from torchvision import transforms
from PIL import Image
import glob


# 将模型加载移出处理函数
def lowlight(image_path, model, device):
    # 1. 读取图片
    data_lowlight = Image.open(image_path).convert('RGB')
    data_lowlight = (np.asarray(data_lowlight) / 255.0)
    data_lowlight = torch.from_numpy(data_lowlight).float()
    # 在进入模型前打印尺寸
    print(f" 图片尺寸: {data_lowlight.shape}", end=" ")

    # 2. 维度变换
    data_lowlight = data_lowlight.permute(2, 0, 1)
    data_lowlight = data_lowlight.to(device).unsqueeze(0)

    # 3. 推理 (计时只包含这里)
    start = time.time()
    _, enhanced_image, _ = model(data_lowlight)
    end_time = (time.time() - start)
    # print(f"推理耗时: {end_time:.4f}秒") # 如果嫌刷屏可以注释掉

    # 4. 保存结果
    file_name = os.path.basename(image_path)
    result_dir = 'data/result_wucunfa_low_exposure_Epoch10/'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    result_path = os.path.join(result_dir, file_name)
    torchvision.utils.save_image(enhanced_image, result_path)
    return end_time


if __name__ == '__main__':
    # --- 阶段 1: 初始化与加载 (只做一次) ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"正在使用设备: {device}")

    print("正在加载模型...")
    DCE_net = model.enhance_net_nopool().to(device)
    DCE_net.load_state_dict(torch.load('snapshts_low_exposure/Epoch10.pth', map_location=device))
    DCE_net.eval()  # 开启评估模式
    print("模型加载完成！")

    # --- 阶段 2: GPU 预热 (关键！) ---
    # 随便造一个假数据跑一次，让 GPU 完成初始化，防止第一张图慢
    print("正在预热 GPU...")
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 512, 512).to(device)
        DCE_net(dummy_input)
    print("预热完成，开始批量处理...")

    # --- 阶段 3: 批量处理 ---
    filePath = 'data/wucunfa_test'
    test_list = glob.glob(os.path.join(filePath, "*"))

    if len(test_list) == 0:
        print(f"警告：在路径 {filePath} 下没有找到任何文件！")

    total_time = 0
    count = 0

    with torch.no_grad():
        for image in test_list:
            if image.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                print(f"正在处理: {os.path.basename(image)}", end=" -> ")

                # 将加载好的 model 和 device 传进去
                t = lowlight(image, DCE_net, device)

                print(f"耗时: {t:.4f}s")
                total_time += t
                count += 1

    if count > 0:
        print(f"\n处理完成！平均每张推理耗时: {total_time / count:.4f} 秒")
    else:
        print("未处理任何图片。")