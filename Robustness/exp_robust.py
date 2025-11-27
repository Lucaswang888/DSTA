from src.DataLoader import DADDataset, CrashDataset
from src.Models import DSTA
from src.eval_tools import evaluation_P_R80
from Robustness.utils_pgd import PGD_DSTA
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import copy
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore')

class Exp_DSTA_Robust:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
        self.model = self._build_model().to(self.device)
        self.checkpoints_path = os.path.join(args.output_dir, args.dataset, 'checkpoints')
        if not os.path.exists(self.checkpoints_path):
            os.makedirs(self.checkpoints_path)

    def _build_model(self):
        # 根据 DataLoader 中的逻辑确定 feature_dim
        if self.args.feature_name == 'vgg16':
            x_dim = 4096
        elif self.args.feature_name == 'res101':
            x_dim = 2048
        else:
            x_dim = 4096 

        # 初始化 DSTA 模型
        model = DSTA(
            x_dim=x_dim,
            h_dim=self.args.hidden_dim,
            z_dim=self.args.latent_dim,
            n_layers=self.args.num_rnn,
            n_obj=19, # 假设是 DAD/Crash 的默认配置
            n_frames=100 if self.args.dataset == 'dad' else 50,
            fps=20.0 if self.args.dataset == 'dad' else 10.0,
            with_saa=True
        )
        return model

    def _get_data(self, flag):
        data_path = os.path.join(self.args.data_path, self.args.dataset)
        
        if self.args.dataset == 'dad':
            phase_map = {'train': 'training', 'val': 'testing', 'test': 'testing'}
            dataset = DADDataset(data_path, self.args.feature_name, phase_map[flag], toTensor=True, device=self.device)
        elif self.args.dataset == 'crash':
            dataset = CrashDataset(data_path, self.args.feature_name, flag, toTensor=True, device=self.device)
        else:
            raise NotImplementedError

        shuffle = True if flag == 'train' else False
        # Drop last fix to avoid batch norm issues if any, though DSTA uses RNN
        data_loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=shuffle, drop_last=True)
        return dataset, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.base_lr)
        return model_optim

    def test_noise(self, test_loader, stddev=0.1):
        """
        在输入特征上添加高斯噪声进行测试
        """
        all_pred = []
        all_labels = []
        all_toas = []
        
        self.model.eval()
        with torch.no_grad():
            for i, batch_data in enumerate(test_loader):
                if len(batch_data) >= 3:
                     batch_x, batch_y, batch_toa = batch_data[0], batch_data[1], batch_data[2]
                
                # 添加噪声
                noise = torch.normal(mean=0.0, std=stddev, size=batch_x.size()).to(self.device)
                batch_x_noisy = batch_x + noise
                
                # Forward
                losses, all_outputs, hiddens, alphas = self.model(
                    batch_x_noisy, batch_y, batch_toa, nbatch=len(test_loader), testing=True
                )

                # 处理输出用于评估 (参考 main.py)
                num_frames = batch_x.size()[1]
                batch_size = batch_x.size()[0]
                pred_frames = np.zeros((batch_size, num_frames), dtype=np.float32)
                
                for t in range(num_frames):
                    pred = all_outputs[t]
                    pred = pred.cpu().numpy() if pred.is_cuda else pred.detach().numpy()
                    pred_frames[:, t] = np.exp(pred[:, 1]) / np.sum(np.exp(pred), axis=1)

                all_pred.append(pred_frames)
                label_onehot = batch_y.cpu().numpy()
                label = np.reshape(label_onehot[:, 1], [batch_size,])
                all_labels.append(label)
                toas = np.squeeze(batch_toa.cpu().numpy()).astype(int)
                all_toas.append(toas)

        all_pred = np.vstack((np.vstack(all_pred[:-1]), all_pred[-1]))
        all_labels = np.hstack((np.hstack(all_labels[:-1]), all_labels[-1]))
        all_toas = np.hstack((np.hstack(all_toas[:-1]), all_toas[-1]))
        
        # 使用 eval_tools 中的评估函数
        dataset_fps = 20.0 if self.args.dataset == 'dad' else 10.0
        AP, mTTA, TTA_R80, P_R80 = evaluation_P_R80(all_pred, all_labels, all_toas, fps=dataset_fps)
        
        print(f"Noise Test (std={stddev}) | AP: {AP:.4f}, mTTA: {mTTA:.4f}")
        return AP

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        test_data, test_loader = self._get_data(flag='test')

        # 加载预训练模型作为 Teacher
        path = os.path.join(self.args.output_dir, self.args.dataset, 'snapshot')
        pretrain_path = self.args.model_file # 用户指定的预训练模型路径
        
        if os.path.exists(pretrain_path):
            print(f"Loading pretrained Teacher model from {pretrain_path}")
            checkpoint = torch.load(pretrain_path, map_location=self.device)
            # 兼容保存格式 (可能是 dict 包含 model state)
            if 'model' in checkpoint:
                self.model.load_state_dict(checkpoint['model'])
            else:
                self.model.load_state_dict(checkpoint)
        else:
            print(f"Warning: Pretrained model not found at {pretrain_path}. Training from scratch might vary.")

        # 创建 Teacher Model (冻结)
        teacher_model = copy.deepcopy(self.model)
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False

        model_optim = self._select_optimizer()
        
        # 定义额外的 Loss
        loss_rob = nn.MSELoss() # 用于特征和输出的一致性回归
        
        # PGD 攻击实例
        pgd_attack = PGD_DSTA(self.model, eps=self.args.eps, alpha=self.args.alpha, steps=self.args.steps, device=self.device)

        print(f"Start Robust Training... Epochs: {self.args.epoch}")
        
        for epoch in range(self.args.epoch):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            
            for i, (batch_x, batch_y, batch_toa) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                
                # 1. 生成对抗样本 (PGD)
                # 使用当前模型生成针对性的对抗样本
                perturbed_batch_x = pgd_attack(batch_x, batch_y, batch_toa)
                
                # 2. Forward pass: Student on Adversarial Input
                losses_pgd, outputs_pgd, hiddens_pgd, _ = self.model(perturbed_batch_x, batch_y, batch_toa, nbatch=len(train_loader))
                
                # 3. Forward pass: Student on Clean Input
                losses_nos, outputs_nos, hiddens_nos, _ = self.model(batch_x, batch_y, batch_toa, nbatch=len(train_loader))
                
                # 4. Forward pass: Teacher on Clean Input
                with torch.no_grad():
                    losses_th, outputs_th, hiddens_th, _ = teacher_model(batch_x, batch_y, batch_toa, nbatch=len(train_loader))

                # --- 计算 Loss ---
                
                # A. 原始任务 Loss (在 PGD 样本上)
                # DSTA 内部计算了 CrossEntropy 和 AuxLoss
                loss_task = losses_pgd['total_loss'].mean() # 确保是标量

                # B. 对抗鲁棒性 Loss (Consistency Loss)
                # Student(Adv) 输出 应该接近 Student(Clean) 输出
                # DSTA 输出是 list of tensors (frames)，我们需要 stack 起来比较
                stack_out_pgd = torch.stack(outputs_pgd) # [frames, batch, 2]
                stack_out_nos = torch.stack(outputs_nos)
                stack_out_th = torch.stack(outputs_th)
                
                adv_loss = loss_rob(stack_out_pgd, stack_out_nos) * self.args.adv_loss_weight

                # C. 蒸馏 Loss (Similarity Loss)
                # Student(Clean) 输出 应该接近 Teacher(Clean) 输出
                sim_loss = loss_rob(stack_out_th, stack_out_nos) * self.args.sim_loss_weight

                # D. 特征 Loss (Feature Matching)
                # Student(Clean) 特征 应该接近 Teacher(Clean) 特征
                # DSTA 返回 hiddens 是一个 list，我们取最后一帧或者所有帧的 stack
                stack_hidden_pgd = torch.stack(hiddens_pgd) # [frames, batch, hidden_dim]
                stack_hidden_nos = torch.stack(hiddens_nos)
                stack_hidden_th = torch.stack(hiddens_th)
                
                # 可以选择只让 Clean Student 逼近 Clean Teacher，也可以让 PGD Student 逼近 Clean Teacher
                # 这里参考框架：Student(PGD) feature vs Student(Clean) feature (Feature Robustness)
                loss_feature = loss_rob(stack_hidden_pgd, stack_hidden_nos) * self.args.feat_loss_weight

                # 总 Loss
                loss = loss_task + adv_loss + sim_loss + loss_feature
                
                train_loss.append(loss.item())
                
                loss.backward()
                model_optim.step()

                if (i + 1) % 50 == 0:
                    print(f"\tEpoch: {epoch + 1}, Iter: {i + 1} | Loss: {loss.item():.5f} (Task: {loss_task:.4f}, Adv: {adv_loss:.4f}, Sim: {sim_loss:.4f})")

            print("Epoch: {} cost time: {:.2f}s".format(epoch + 1, time.time() - epoch_time))
            avg_train_loss = np.average(train_loss)
            print(f"Epoch: {epoch+1} | Avg Train Loss: {avg_train_loss:.7f}")

            # 验证 (在干净数据上 + 噪声数据上)
            # print("Validating on Clean Data...")
            # self.test_noise(test_loader, stddev=0.0) # Clean
            # print("Validating on Noisy Data (std=0.2)...")
            # self.test_noise(test_loader, stddev=0.2)

            # 保存模型
            save_path = os.path.join(self.checkpoints_path, f'rob_checkpoint_epoch_{epoch}.pth')
            torch.save(self.model.state_dict(), save_path)
        
        # 保存最终最好的 (这里简化为最后一个)
        final_path = os.path.join(self.checkpoints_path, 'rob_final.pth')
        torch.save(self.model.state_dict(), final_path)
        return self.model
