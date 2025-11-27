import argparse
import os
import torch
import numpy as np
from Robustness.exp_robust import Exp_DSTA_Robust

# 设置随机种子
seed = 123
np.random.seed(seed)
torch.manual_seed(seed)

def main():
    parser = argparse.ArgumentParser(description='DSTA Robustness Experiment')

    # Basic Config
    parser.add_argument('--dataset', type=str, default='dad', choices=['dad', 'crash'], help='Dataset name')
    parser.add_argument('--data_path', type=str, default='./data', help='Data path')
    parser.add_argument('--feature_name', type=str, default='vgg16', choices=['vgg16', 'res101'], help='Feature type')
    parser.add_argument('--model_file', type=str, required=True, help='Path to the PRETRAINED model for teacher initialization')
    parser.add_argument('--output_dir', type=str, default='./output_robust', help='Output directory')
    
    # Model Config (必须与预训练模型一致)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--num_rnn', type=int, default=1)
    
    # Training Config
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--epoch', type=int, default=10, help='Robust finetuning epochs')
    parser.add_argument('--base_lr', type=float, default=1e-5, help='Lower LR for finetuning')
    
    # Robustness / Attack Config
    parser.add_argument('--eps', type=float, default=0.01, help='PGD epsilon (perturbation magnitude)')
    parser.add_argument('--alpha', type=float, default=0.002, help='PGD step size')
    parser.add_argument('--steps', type=int, default=5, help='PGD steps')
    
    # Loss Weights
    parser.add_argument('--adv_loss_weight', type=float, default=1.0, help='Weight for consistency loss (Adv vs Clean)')
    parser.add_argument('--sim_loss_weight', type=float, default=1.0, help='Weight for distillation loss (Student vs Teacher)')
    parser.add_argument('--feat_loss_weight', type=float, default=0.1, help='Weight for feature matching loss')
    
    args = parser.parse_args()

    # 运行实验
    exp = Exp_DSTA_Robust(args)
    
    # 1. 开始鲁棒性训练
    print(">>>>>>> Start Robustness Training >>>>>>>>>>>>>>>>>>>>>>>>>>")
    exp.train(setting='robust_finetune')
    
    # 2. 训练后测试
    print(">>>>>>> Testing Robustness against Noise >>>>>>>>>>>>>>>>>>>")
    _, test_loader = exp._get_data(flag='test')
    
    print("Test on Clean Data:")
    exp.test_noise(test_loader, stddev=0.0)
    
    print("Test on Noisy Data (std=0.1):")
    exp.test_noise(test_loader, stddev=0.1)
    
    print("Test on Noisy Data (std=0.2):")
    exp.test_noise(test_loader, stddev=0.2)
    
    print("Test on Noisy Data (std=0.3):")
    exp.test_noise(test_loader, stddev=0.3)

if __name__ == "__main__":
    main()
