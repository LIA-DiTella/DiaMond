import os
import random
import wandb
import torch
from torch import nn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from sklearn.metrics import confusion_matrix, roc_auc_score, balanced_accuracy_score, f1_score
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import precision_score, recall_score
from DiaMond import DiaMond, Head
from adni import AdniDataset
from optimizer import LARS, CosineWarmupScheduler
from regbn import RegBN


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

print(f"Torch: {torch.__version__}")
print(f"Cuda Available: {torch.cuda.is_available()}")

##################################################################################
#functions
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@torch.no_grad()
def init_weights(m: torch.Tensor):
    if isinstance(m, (nn.Conv3d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.BatchNorm3d)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

def get_output(regbn_module, batch_data, model, model_pet, model_mri, model_mp, head, 
            device, modality = None, steps_per_epoch: int=-1, epoch_id: int=-1, is_training: bool=False):
    (mri_data, pet_data), label = batch_data 
    mri_data, pet_data, label = mri_data.to(device), pet_data.to(device), label.to(device)
    data = (mri_data, pet_data)
    if modality == "multi":
        output_pet = model_pet(pet_data)
        output_mri = model_mri(mri_data)
        
        # apply late regbn
        if is_training: 
            kwargs_regbn_train = {"is_training": True, 'n_epoch': epoch_id, 'steps_per_epoch': steps_per_epoch}
        else:
            kwargs_regbn_train = {"is_training": False}
        
        output_pet, output_mri = regbn_module(output_pet, output_mri, **kwargs_regbn_train)

        output_mp = model_mp(pet_data, mri_data)

        output = (output_pet + output_mri + output_mp) / 3
    else:
        raise ValueError(f"Modality {modality} not implemented")

    if head is not None:
        output = head(output)
    return output, label
##################################################################################

#Train function
def train(regbn_module, model, head, optimizer, dataloader, num_classes, 
        loss_fn = nn.BCEWithLogitsLoss(), modality = None, epoch_id: int=None):
    
    model_pet, model_mri, model_mp = model
    model_pet, model_mri, model_mp = model_pet.train(), model_mri.train(), model_mp.train()

    if head is not None:
        head.train()
    
    cumulative_loss = 0.0
    cumulative_acc = 0.0
    all_predicted_class_labels = np.asarray([])
    all_loss_labels = np.asarray([])
    steps_per_epoch = len(dataloader)
    for batch_data in dataloader:      
        output, label = get_output(regbn_module, batch_data, model, model_pet, model_mri, model_mp, head, device, modality,
            steps_per_epoch=steps_per_epoch, 
            epoch_id=epoch_id,
            is_training=True,)      
        loss = loss_fn(output.squeeze(1).float(), label.float()) if num_classes == 2 else loss_fn(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
  
        if num_classes == 2:
            predicted_class_labels = output.squeeze(1)
            predicted_class_labels = torch.nn.Sigmoid()(predicted_class_labels)
            predicted_class_labels = torch.round(predicted_class_labels)
        else:
            predicted_class_labels = torch.argmax(output, dim=1)

        all_predicted_class_labels = np.append(all_predicted_class_labels, predicted_class_labels.detach().cpu())
        all_loss_labels = np.append(all_loss_labels, label.detach().cpu())    
        cumulative_loss += loss.item()
    acc = balanced_accuracy_score(all_loss_labels.flatten(), all_predicted_class_labels.flatten(), adjusted=False) \
        if num_classes == 2 else balanced_accuracy_score(all_loss_labels, all_predicted_class_labels, adjusted=False)
    return cumulative_loss / len(dataloader), acc
    
############################################################################################

#Validation function
def calculate_val_loss(regbn_module, model, head, dataloader, num_classes, loss_fn = nn.BCEWithLogitsLoss(), modality = None):

    model_pet, model_mri, model_mp = model
    model_pet, model_mri, model_mp = model_pet.eval(), model_mri.eval(), model_mp.eval()

    if head is not None:
        head.eval()
    cumulative_loss = 0.0
    cumulative_acc = 0.0
    all_predicted_class_labels = np.asarray([])
    all_loss_labels = np.asarray([])

    with torch.no_grad():
        for batch_data in dataloader:
            output, label = get_output(regbn_module, batch_data, model, model_pet, model_mri, model_mp, head, device, modality)
            loss = loss_fn(output.squeeze(1).float(), label.float()) if num_classes == 2 else loss_fn(output, label)
            cumulative_loss += loss.item()
            
            if num_classes == 2: 
                predicted_class_labels = output.squeeze(1)
                predicted_class_labels = torch.nn.Sigmoid()(predicted_class_labels)
                predicted_class_labels = torch.round(predicted_class_labels)
            else:
                predicted_class_labels = torch.argmax(output, dim=1)

            all_predicted_class_labels = np.append(all_predicted_class_labels, predicted_class_labels.int().detach().cpu())
            all_loss_labels = np.append(all_loss_labels, label.int().detach().cpu())
            
    acc = balanced_accuracy_score(all_loss_labels.flatten(), all_predicted_class_labels.flatten(), adjusted=False) \
        if num_classes == 2 else balanced_accuracy_score(all_loss_labels, all_predicted_class_labels, adjusted=False)
    cm = confusion_matrix(all_loss_labels.flatten(), all_predicted_class_labels.flatten()) \
        if num_classes == 2 else confusion_matrix(all_loss_labels, all_predicted_class_labels)
    f1 = f1_score(all_loss_labels.flatten(), all_predicted_class_labels.flatten(), average='binary') \
        if num_classes == 2 else f1_score(all_loss_labels, all_predicted_class_labels, average='macro')

    return cumulative_loss / len(dataloader), acc, cm, f1
    
############################################################################################

#Test function
def test(regbn_module, model, head, dataloader, num_classes, loss_fn = nn.BCEWithLogitsLoss(), modality = None):

    model_pet, model_mri, model_mp = model
    model_pet, model_mri, model_mp = model_pet.eval(), model_mri.eval(), model_mp.eval()

    if head is not None:
        head.eval()
    cumulative_loss = 0.0
    cumulative_acc = 0.0
    all_predicted_class_scores = np.asarray([])
    all_predicted_class_labels = np.asarray([])
    all_loss_labels = np.asarray([])

    with torch.no_grad():
        for batch_data in dataloader:
            output, label = get_output(regbn_module, batch_data, model, model_pet, model_mri, model_mp, head, device, modality)
            loss = loss_fn(output.squeeze(1).float().to(device), label.float().to(device)) if num_classes == 2 else loss_fn(output.to(device), label.to(device))
            cumulative_loss += loss.item()
          
            if num_classes == 2:
                predicted_class_scores = output.squeeze(1)
                predicted_class_labels = torch.nn.Sigmoid()(predicted_class_scores)
                predicted_class_labels = torch.round(predicted_class_labels)
            else:
                predicted_class_scores = output
                predicted_class_labels = torch.argmax(output, dim=1)
        
            all_predicted_class_scores = np.append(all_predicted_class_scores, predicted_class_scores.int().detach().cpu())
            all_predicted_class_labels = np.append(all_predicted_class_labels, predicted_class_labels.int().detach().cpu())
            all_loss_labels = np.append(all_loss_labels, label.int().detach().cpu())
            
    acc = balanced_accuracy_score(all_loss_labels.flatten(), all_predicted_class_labels.flatten(), adjusted=False) \
        if num_classes == 2 else balanced_accuracy_score(all_loss_labels, all_predicted_class_labels, adjusted=False)
    cm = confusion_matrix(all_loss_labels.flatten(), all_predicted_class_labels.flatten()) \
        if num_classes == 2 else confusion_matrix(all_loss_labels, all_predicted_class_labels)
    f1 = f1_score(all_loss_labels.flatten(), all_predicted_class_labels.flatten(), average='binary') \
        if num_classes == 2 else f1_score(all_loss_labels, all_predicted_class_labels, average='macro')
    auc = roc_auc_score(all_loss_labels.flatten(), all_predicted_class_scores.flatten()) \
        if num_classes == 2 else 0
    
    # Precision and Recall calculation
    precision = precision_score(all_loss_labels.flatten(), all_predicted_class_labels.flatten(), average='binary') \
        if num_classes == 2 else precision_score(all_loss_labels, all_predicted_class_labels, average='micro')
    recall = recall_score(all_loss_labels.flatten(), all_predicted_class_labels.flatten(), average='binary') \
        if num_classes == 2 else recall_score(all_loss_labels, all_predicted_class_labels, average='micro')
    
    return cumulative_loss / len(dataloader), acc, cm, f1, auc, precision, recall
    
############################################################################################

if __name__ == '__main__':
    cuda_id = 0
    device = torch.device(f'cuda:{cuda_id}' if torch.cuda.is_available() else 'cpu')
    
    with open('../config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    experiment_name = 'mri+pet'

    test_bacc = []
    test_f1 = []
    test_auc = []
    test_cm = []
    test_precision = []
    test_recall = []

    regbn_kwargs = {
        'gpu': cuda_id,
        'f_num_channels': 192 if config['class_num'] == 3 else 128, 
        'g_num_channels': 192 if config['class_num'] == 3 else 128, 
        'f_layer_dim': [],
        'g_layer_dim': [],
        'normalize_input': True, 
        'normalize_output': True,
        'affine': True,
        'sigma_THR': 0.0, 
        'sigma_MIN': 0.0, 
    }

    for split in range(0, 5):
        run = wandb.init(
            project="DiaMond",
            entity="",
            notes=experiment_name,
            tags=[],
            config=config,
            mode='disabled',
            )
        seed_everything(wandb.config.seed)
        dataset_path = wandb.config.dataset_path

        if not (wandb.config.test):
            train_data = AdniDataset(path=f'{dataset_path}/{split}-train.h5', is_training=True, 
                        out_class_num=wandb.config.class_num, with_mri=wandb.config.with_mri, with_pet=wandb.config.with_pet)
            valid_data = AdniDataset(path=f'{dataset_path}/{split}-valid.h5', is_training=False,
                        out_class_num=wandb.config.class_num, with_mri=wandb.config.with_mri, with_pet=wandb.config.with_pet)
            
            train_loader = DataLoader(dataset = train_data, batch_size=wandb.config.batch_size, shuffle=True, num_workers=12, drop_last=True)
            valid_loader = DataLoader(dataset = valid_data, batch_size=wandb.config.batch_size, shuffle=True, num_workers=4, drop_last=False)
        else:
            test_data = AdniDataset(path=f'{dataset_path}/{split}-test.h5', is_training=False,
                    out_class_num=wandb.config.class_num, with_mri=wandb.config.with_mri, with_pet=wandb.config.with_pet)
            test_loader = DataLoader(dataset = test_data, batch_size=wandb.config.batch_size, shuffle=True)

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        wandb.config.update( {'device': device}, allow_val_change=True)

        head = None
        
        if wandb.config.model == "DiaMond":
            diamond = DiaMond()
            model = diamond.body_all(
                # PATH_PET = f'path/to/mono_pet.pt',
                # PATH_MRI = f'path/to/mono_mri.pt',
                modality = wandb.config.modality,
                block_size = wandb.config.block_size,
                image_size = wandb.config.img_size,
                patch_size = wandb.config.patch_size,
                num_classes = wandb.config.class_num,
                channels = wandb.config.in_chans,
                dim = wandb.config.dim,
                depth = wandb.config.depth,
                heads = wandb.config.heads,
                dropout = wandb.config.dropout,
                mlp_dim = 309
            )
            head = diamond.head(
                block_size = wandb.config.block_size,
                image_size = wandb.config.img_size,
                num_classes = wandb.config.class_num,
                channels = wandb.config.in_chans,
            )

            for param in model[0].parameters():
                param.requires_grad = True 
            for param in model[1].parameters():
                param.requires_grad = True 
            for param in model[2].parameters():
                param.requires_grad = True
            
            regbn_module = RegBN(**regbn_kwargs).to(device)
        
        else:
            raise ValueError(f"Model {wandb.config.model} not implemented")

        if wandb.config.reweight:
            cls_weight = torch.tensor([0.7226, 0.9394, 1.8129]).to(device)
            # cls_weight = torch.tensor([1.0, 1.0, 10.0]).to(device)
        else:
            cls_weight = None
        
        loss_fn = nn.BCEWithLogitsLoss() if wandb.config.class_num == 2 else nn.CrossEntropyLoss(weight=cls_weight)
      
        if not (wandb.config.test):
            pytorch_total_params = sum(p.numel() for m in model for p in m.parameters())
            pytorch_total_train_params = sum(p.numel() for m in model for p in m.parameters() if p.requires_grad)
            if head is not None:
                pytorch_total_params += sum(p.numel() for p in head.parameters())
                pytorch_total_train_params += sum(p.numel() for p in head.parameters() if p.requires_grad)
            print(f'total params: {pytorch_total_params}')        
            print(f'total trainable params: {pytorch_total_train_params}')

            if wandb.config.pretrained is not None:
                pretrained = torch.load(wandb.config.pretrained)
                msg = [m.load_state_dict(pretrained['model_state_dict'][i]) for i, m in enumerate(model)]
                print('load pretrained model:', msg)
            else:
                pretrained = None
                model[-1].apply(init_weights)
            
            model = [m.to(device) for m in model]

            optimizer_params = [p for m in model for p in m.parameters() if p.requires_grad]

            if head is not None:
                if pretrained is not None:
                    msg = head.load_state_dict(pretrained['head_state_dict'])
                    print('load pretrained head:', msg)
                else:
                    head.apply(init_weights)

                head = head.to(device)
                optimizer_params += [p for p in head.parameters() if p.requires_grad]

            if wandb.config.optimizer == "sgd":
                optimizer = torch.optim.SGD(optimizer_params,
                                    lr=wandb.config.lr, momentum=wandb.config.momentum, weight_decay=wandb.config.weight_decay)
            elif wandb.config.optimizer == "adam":
                optimizer = torch.optim.Adam(optimizer_params,
                                    lr=wandb.config.lr)
            elif wandb.config.optimizer == "adamW":
                optimizer = torch.optim.AdamW(optimizer_params,
                                    lr=wandb.config.lr, weight_decay=wandb.config.weight_decay)
            elif wandb.config.optimizer == "LARS":
                optimizer = LARS(optimizer_params,
                                    lr=wandb.config.lr, weight_decay=wandb.config.weight_decay)
                
            if wandb.config.scheduler == "StepLR":
                scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
            elif wandb.config.scheduler == "ReduceLROnPlateau":
                scheduler = ReduceLROnPlateau(optimizer, 'min')
            elif wandb.config.scheduler == "CosineAnnealingLR":
                scheduler = CosineWarmupScheduler(optimizer, max_steps=wandb.config.epochs, 
                        warmup_steps=wandb.config.warmup_steps, lr=wandb.config.lr, batch_size=wandb.config.batch_size, end_lr=0.00001)

            if pretrained is not None:
                optimizer.load_state_dict(pretrained['optimizer_state_dict'])
                init_epoch = pretrained['epoch']
            else:
                init_epoch = 1      
            
            print(f"======= Starting Training {wandb.config.model}, {wandb.config.modality} from Epoch {init_epoch} =========")
            # wandb.watch(model, log='all', log_freq=10)
            # if head is not None:
            #     wandb.watch(head, log='all', log_freq=10)
            
            max_accuracy = 0.0
            max_epoch = 0
            for epoch in range(init_epoch, wandb.config.epochs + 1):
                loss, acc = train(regbn_module, model, head, optimizer, train_loader, 
                        wandb.config.class_num, loss_fn, modality = wandb.config.modality, epoch_id=epoch)
                print(optimizer.param_groups[0]["lr"])
                
                if wandb.config.scheduler == "StepLR":
                    scheduler.step()
                elif wandb.config.scheduler == "ReduceLROnPlateau":
                    scheduler.step(loss)
                elif wandb.config.scheduler == "CosineAnnealingLR":
                    scheduler.step()

                val_loss, val_acc, val_cm, val_f1 = calculate_val_loss(regbn_module, model, head, valid_loader, wandb.config.class_num, loss_fn, modality = wandb.config.modality)
                
                wandb.log({'train_loss': loss, 'train_bacc': acc, 'val_loss': val_loss, 'val_bacc': val_acc, 'epoch': epoch, 'lr': optimizer.param_groups[0]["lr"]})
                print(f"train_loss: {loss}, train_acc: {acc}, val_loss: {val_loss}, val_acc: {val_acc}, val_cm: {val_cm}, val_f1: {val_f1}, epoch: {epoch}")

                
                save_folder = f'../models/{wandb.config.model}/{experiment_name}'
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder, exist_ok=True)
                with open(f'{save_folder}/_hyperparams.yaml', 'w') as outfile:
                    yaml.safe_dump(config, outfile, default_flow_style=False)
                
                if max_accuracy < val_acc:
                    max_accuracy = val_acc
                    max_epoch = epoch
                    if wandb.config.save:
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': [m.state_dict() for m in model],
                            'head_state_dict': head.state_dict() if head is not None else None,
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss
                            }, f'{save_folder}/{wandb.config.model}_{wandb.config.modality}_split{split}_bestval.pt')
                
                if wandb.config.save and epoch % 10 == 0:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': [m.state_dict() for m in model],
                        'head_state_dict': head.state_dict() if head is not None else None,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss
                        }, f'{save_folder}/{wandb.config.model}_{wandb.config.modality}_split{split}_latest.pt')

            wandb.log({'max_bacc': max_accuracy, 'max_bacc_epoch': max_epoch})
            print(f"Max BACC: {max_accuracy} at epoch {max_epoch}")
            print(f"****************************Done with split {split}****************************")

        else:
            save_folder = f'../models/{wandb.config.model}/{experiment_name}'
            results_folder = save_folder
            if not os.path.exists(results_folder):
                os.makedirs(results_folder, exist_ok=True)
            with open(f'{results_folder}/_hyperparams.yaml', 'w') as outfile:
                yaml.safe_dump(config, outfile, default_flow_style=False)
            
            checkpoint = torch.load(f'{save_folder}/{wandb.config.model}_{wandb.config.modality}_split{split}_bestval.pt')
            msg = [m.load_state_dict(checkpoint['model_state_dict'][i]) for i, m in enumerate(model)]
            [m.to(device) for m in model]
            if head is not None:
                msg_head = head.load_state_dict(checkpoint['head_state_dict'])
                head.to(device)
            
            epoch = checkpoint['epoch']
            print(f'========= Loaded model from Epoch: {epoch} =================')
            print(msg)
            test_loss, test_accuracy, cm, f1, auc, precision, recall = test(regbn_module, model, head, test_loader, wandb.config.class_num, loss_fn, modality = wandb.config.modality)
            
            test_bacc.append(test_accuracy)
            test_f1.append(f1)
            test_auc.append(auc)
            test_cm.append(cm)
            test_precision.append(precision)
            test_recall.append(recall)

            wandb.log({'test_loss': test_loss, 'test_accuracy': test_accuracy, 'cm': cm, 'epoch': epoch})
            cm = str(cm).replace('\n', ' ')
            print(f"epoch: {epoch}: test_loss: {test_loss}, test_bacc: {test_accuracy}, cm: {cm}, f1: {f1}, auc: {auc}, precision: {precision}, recall: {recall}")
            with open(f'{results_folder}/{wandb.config.model}_{wandb.config.modality}_test.txt', 'a') as f:
                f.write(f"Split-{split}: test_bacc: {test_accuracy}, f1: {f1}, auc: {auc}, cm: {cm}, precision: {precision}, recall: {recall}\n")

    if wandb.config.test:

        cm = np.sum(np.array(test_cm), axis=0)
        print(f'=======Finished Testing with Average of splits=======')
        wandb.log({'test_bacc': (np.mean(test_bacc), np.std(test_bacc)), 'test_f1':(np.mean(test_f1), np.std(test_f1)), 
                    'test_auc': (np.mean(test_auc), np.std(test_auc)), 'test_precision': (np.mean(test_precision), np.std(test_precision)), 
                    'test_recall': (np.mean(test_recall), np.std(test_recall))})
        
        cm = str(cm).replace('\n', ' ')
        print(f"test_bacc: {np.mean(test_bacc):.6f} +- {np.std(test_bacc):.6f}, \
                test_f1: {np.mean(test_f1):.6f} +- {np.std(test_f1):.6f}, \
                test_auc: {np.mean(test_auc):.6f} +- {np.std(test_auc):.6f}, cm: {cm}, \
                test_precision: {np.mean(test_precision):.6f} +- {np.std(test_precision):.6f}, \
                test_recall: {np.mean(test_recall):.6f} +- {np.std(test_recall):.6f}")
        
        with open(f'{results_folder}/{wandb.config.model}_{wandb.config.modality}_test.txt', 'a') as f:
            f.write(f"Mean test_bacc: {np.mean(test_bacc):.6f} +- {np.std(test_bacc):.6f}, \
            test_f1: {np.mean(test_f1):.6f} +- {np.std(test_f1):.6f}, \
            test_auc: {np.mean(test_auc):.6f} +- {np.std(test_auc):.6f}, cm: {cm}, \
            test_precision: {np.mean(test_precision):.6f} +- {np.std(test_precision):.6f}, \
            test_recall: {np.mean(test_recall):.6f} +- {np.std(test_recall):.6f}\n\n")


