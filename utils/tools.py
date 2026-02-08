import numpy as np
import torch
import matplotlib.pyplot as plt
import shutil
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
import torch.nn.functional as F
from scipy.signal.windows import gaussian
from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error
import time
from torch import nn
import wandb
plt.switch_backend('agg')
def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num, 'Percent': trainable_num/total_num}

class Augment_time_series_family(object):
    '''
    This is a set of augmentation for methods for time series
    '''
    def __init__(self, n_holes, mean=0, std=0.02):
        pass

class Downsample_Expand_aug(object):
    '''
    '''
    def __init__(self, rate=0.1):
        pass

class Masking_aug(object):
    def __init__(self, drop_rate):
        self.drop_rate = drop_rate
    
    def __call__(self, seq):
        '''
        Params:
            seq: Tensor sequence of size (B, num_var, L)
        '''
        seq = F.dropout(seq, self.drop_rate)
        return seq
    
    
def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.

    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

def adjust_learning_rate(accelerator, optimizer, scheduler, epoch, args, printout=True):
    if args.lradj == 'type1':
        # lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
        if epoch > args.least_epochs:
            lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - args.least_epochs) // 1))}
        else:
            lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'PEMS':
        lr_adjust = {epoch: args.learning_rate * (0.95 ** (epoch // 1))}
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout:
            if accelerator is not None:
                accelerator.print(f'{args.lradj}| Updating learning rate to {lr}')
            else:
                print(f'{args.lradj}| Updating learning rate to {lr}')


class EarlyStopping:
    def __init__(self, args, accelerator=None, patience=7, verbose=False, delta=0, save_mode=True, least_epochs=5):
        self.args = args
        self.accelerator = accelerator
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_vali_mae = None
        self.best_test_mae = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.save_mode = save_mode
        self.least_epochs = least_epochs

    def __call__(self, epoch, val_loss, vali_mae_loss, test_mae_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            if self.save_mode:
                self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            if epoch > self.least_epochs:
                # the early stopping won't count before some epoches are trained
                self.counter += 1
            if self.accelerator is None:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            else:
                self.accelerator.print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            # self.best_vali_mae = vali_mae_loss
            # self.best_test_mae = test_mae_loss
            if self.save_mode:
                self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            if self.accelerator is not None:
                self.accelerator.print(
                    f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            else:
                print(
                    f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        if self.accelerator is not None:
            
            # model = self.accelerator.unwrap_model(model)
            # self.accelerator.save(model.state_dict(), path + '/' + 'checkpoint')
            # self.accelerator.wait_for_everyone()
            # if self.args.use_LoRA:
            #     model.save_pretrained(path, save_adapter=True, save_config=True)
            #     self.accelerator.save_model(model, path)
            # else:
            #     self.accelerator.save_model(model, path)
            self.accelerator.save_model(model, path)
            # self.accelerator.save_state(path)
            #self.accelerator.save(model, path + '/' + 'checkpoint.pth')
            self.accelerator.print(f'The checkpoint is saved in {path}!')
            # self.accelerator.save_state(path + '/')
        else:
            #torch.save(model, path + '/' + 'checkpoint.pth')
            torch.save(model.state_dict(), path + '/' + 'checkpoint')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def del_files(dir_path):
    shutil.rmtree(dir_path, ignore_errors=True)

def vali_baseline(args, accelerator, model, vali_data, vali_loader, criterion, compute_seen_unseen=False):
    total_preds, total_references = [], []
    total_seen_unseen_ids = []
    model.eval()
    with torch.no_grad():
        for i, (cycle_curve_data, curve_attn_mask,  labels, life_class, scaled_life_class, weights, seen_unseen_ids) in tqdm(enumerate(vali_loader)):
            cycle_curve_data = cycle_curve_data.float().to(accelerator.device)# [B, S, N]
            curve_attn_mask = curve_attn_mask.float().to(accelerator.device)
            labels = labels.float().to(accelerator.device)

            # encoder - decoder
            outputs = model(cycle_curve_data, curve_attn_mask)
            # self.accelerator.wait_for_everyone()
            std, mean_value = np.sqrt(vali_data.label_scaler.var_[-1]), vali_data.label_scaler.mean_[-1]
            transformed_preds = outputs * std + mean_value
            transformed_labels = labels * std + mean_value

            all_predictions, all_targets, seen_unseen_ids = accelerator.gather_for_metrics((transformed_preds, transformed_labels, seen_unseen_ids))

         
            total_preds = total_preds + all_predictions.detach().cpu().numpy().reshape(-1).tolist()
            total_references = total_references + all_targets.detach().cpu().numpy().reshape(-1).tolist()
            if compute_seen_unseen:
                total_seen_unseen_ids = total_seen_unseen_ids + seen_unseen_ids.detach().cpu().numpy().reshape(-1).tolist()

    total_preds = np.array(total_preds)
    total_references = np.array(total_references)   
    total_seen_unseen_ids = np.array(total_seen_unseen_ids)
    rmse = root_mean_squared_error(total_references, total_preds)
    mae = mean_absolute_error(total_references, total_preds)
    mape = mean_absolute_percentage_error(total_references, total_preds)

    relative_error = abs(total_preds - total_references) / total_references
    hit_num = sum(relative_error<=args.alpha1)
    alpha_acc1 = hit_num / len(total_references) * 100

    relative_error = abs(total_preds - total_references) / total_references
    hit_num = sum(relative_error<=args.alpha2)
    alpha_acc2 = hit_num / len(total_references) * 100

    if compute_seen_unseen:
        # calculate the model performance on the samples from the seen and unseen aging conditions
        seen_references = total_references[total_seen_unseen_ids==1] if np.any(total_seen_unseen_ids==1) else np.array([0])
        unseen_references = total_references[total_seen_unseen_ids==0] if np.any(total_seen_unseen_ids==0) else np.array([0])
        seen_preds = total_preds[total_seen_unseen_ids==1] if np.any(total_seen_unseen_ids==1) else np.array([1])
        unseen_preds = total_preds[total_seen_unseen_ids==0] if np.any(total_seen_unseen_ids==0) else np.array([1])

        # MAPE
        seen_mape = mean_absolute_percentage_error(seen_references, seen_preds)
        if len(unseen_preds) > 0:
            unseen_mape = mean_absolute_percentage_error(unseen_references, unseen_preds)
        else:
            unseen_mape = -10000

        # alpha-acc1 
        relative_error = abs(seen_preds - seen_references) / seen_references
        hit_num = sum(relative_error<=args.alpha1)
        seen_alpha_acc1 = hit_num / len(seen_references) * 100

        
        if len(unseen_preds) > 0:
            relative_error = abs(unseen_preds - unseen_references) / unseen_references
            hit_num = sum(relative_error<=args.alpha1)
            unseen_alpha_acc1 = hit_num / len(unseen_references) * 100
        else:
            unseen_alpha_acc1 = -10000

        # alpha-acc2
        relative_error = abs(seen_preds - seen_references) / seen_references
        hit_num = sum(relative_error<=args.alpha2)
        seen_alpha_acc2 = hit_num / len(seen_references) * 100

        if len(unseen_preds) > 0:
            relative_error = abs(unseen_preds - unseen_references) / unseen_references
            hit_num = sum(relative_error<=args.alpha2)
            unseen_alpha_acc2 = hit_num / len(unseen_references) * 100
        else:
            unseen_alpha_acc2 = -10000

        model.train()
        return  rmse, mae, mape, alpha_acc1, alpha_acc2, unseen_mape, seen_mape, unseen_alpha_acc1, seen_alpha_acc1, unseen_alpha_acc2, seen_alpha_acc2
    
    model.train()
    return rmse, mae, mape, alpha_acc1, alpha_acc2


def load_content(args):
    if 'ETT' in args.data:
        file = 'ETT'
    else:
        file = args.data
    with open('./dataset/prompt_bank/{0}.txt'.format(file), 'r') as f:
        content = f.read()
    return content