import torch
from dataset import *
import argparse
import wandb
from tqdm import tqdm
from utils import rle_encode, DiceScoreSum
import models as m
import numpy as np
import pandas as pd
import time
import os
from loss import *
from scheduler import CosineAnnealingWarmUpRestarts

def parseargs():
    parser = argparse.ArgumentParser(description="Satellite 0601p zz")

    parser.add_argument('--run_name', type=str, default='Satellite')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--dataset_path', type=str, default="./dataset/")
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--eval_per_epoch', type=int, default=5)
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoint/')
    parser.add_argument('--model_name', type=str, default='UNet')
    parser.add_argument('--threshold', type=float, default=0.35)
    parser.add_argument('--mode', type=str, default='train') # train, submit, threshold
    parser.add_argument('--submit_ckpt', type=str, default='UNet_5')
    parser.add_argument('--continue_learn', type=bool, default=False) # train continue with loaded ckpt
    parser.add_argument('--continue_ckpt', type=str, default='./checkpoint/UNet_5')
    parser.add_argument('--augment', type=bool, default=False) # Train with augmentation dataloader
    parser.add_argument('--loss', type=str, default='torch.nn.BCEWithLogitsLoss') # loss class name
    parser.add_argument('--max_lr', type=float, default=0.001)
    parser.add_argument('--peak_epoch',type=int, default=10)
    parser.add_argument('--lr_period', type=int, default=100)
    parser.add_argument('--lr_gamma', type=float, default=0.7)
    parser.add_argument('--loss_alpha', type=float, default=0.01)
    parser.add_argument('--last_save', type=bool, default=False)
    parser.add_argument('--tau', type=float, default=0.1)
    parser.add_argument('--t_k', type=float, default=50)

    args = parser.parse_args()
    return args

def train(args):
    trainloader, validloader = get_train_valid_loader(args)

    if args.device == 'cuda':
        if not torch.cuda.is_available():
            print("[Error] Cannot use cuda.")
            return -1

    device = args.device

    wandb.init(
        project="[AI-Competetion] Satellite Image Segmentation",
        name=f"{args.model_name}_{time.strftime('%m%d%H%M', time.localtime())}"
    )
    wandb.config.update(args)
    
    os.makedirs(args.ckpt_dir, exist_ok=True)
    
    model = eval('m.' + args.model_name + '().to(device)')
    criterion = eval(args.loss+'()')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    total_steps = len(trainloader) * args.epochs
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=args.lr_period, eta_max=args.max_lr, T_up=args.peak_epoch, gamma=args.lr_gamma)

    if args.continue_learn:
        model.load_state_dict(torch.load(args.continue_ckpt))
    
    # Training
    best_loss = torch.inf
    print("Training Start")
    epochs = tqdm(range(args.epochs), leave=False)
    
    for epoch in epochs:
        train_loss = 0
        
        model.train()
        for images, masks in trainloader:
            images = images.float().to(device)
            masks = masks.float().to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks.unsqueeze(1))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        
        scheduler.step()

        train_loss /= len(trainloader)
        current_lr = scheduler.get_lr()[0]

        print(f"Epoch {epoch+1} / train loss: {train_loss}, lr: {current_lr}")
        wandb.log({"train/loss": train_loss})
        wandb.log({"train/lr": current_lr})

        # Inner evaluation step
        if (epoch+1) % args.eval_per_epoch == 0:
            val_loss = 0
            dice = 0.0
            with torch.no_grad():
                model.eval()
                for images, masks in validloader:
                    images = images.float().to(device)
                    masks = masks.float().to(device)

                    outputs = model(images)
                    loss = criterion(outputs, masks.unsqueeze(1))
                    val_loss += loss.item()
                    pred = (torch.sigmoid(outputs) > args.threshold).squeeze(1)
                    dice += DiceScoreSum(pred, masks)

            dice /= (7140 - int(7140 * 0.95)) * 64 # test data num
            val_loss /= len(validloader)
            print(f"Test loss: {val_loss}")
            print(f"Test Dice Score: {dice}")
            wandb.log({"test/loss": val_loss})
            wandb.log({"test/dice": dice})

            if val_loss < best_loss:
                best_loss = val_loss
                directory = args.ckpt_dir + args.model_name + "_" + str(epoch+1)
                print(F"Valid loss decreased. \nSave model on {directory}")

                torch.save(model.state_dict(), directory)

            if (epoch+1) == args.epochs:
                if args.last_save:
                    best_loss = val_loss
                    directory = args.ckpt_dir + args.model_name + "_" + str(epoch+1)
                    print(F"Save last model on {directory}")

                    torch.save(model.state_dict(), directory)
                
def train_domain(args):
    trainloader, validloader = get_train_valid_loader(args)
    testset = SatelliteTestDataset_No_Ram(dataset_path = args.dataset_path, csv_file = 'test.csv')
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    if args.device == 'cuda':
        if not torch.cuda.is_available():
            print("[Error] Cannot use cuda.")
            return -1

    device = args.device

    wandb.init(
        project="[AI-Competetion] Satellite Image Segmentation",
        name=f"{args.model_name}_{time.strftime('%m%d%H%M', time.localtime())}"
    )
    wandb.config.update(args)
    
    os.makedirs(args.ckpt_dir, exist_ok=True)
    
    model = eval('m.' + args.model_name + '().to(device)')
    criterion = eval(args.loss+'()')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    total_steps = len(trainloader) * args.epochs
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=args.lr_period, eta_max=args.max_lr, T_up=args.peak_epoch, gamma=args.lr_gamma)

    if args.continue_learn:
        model.load_state_dict(torch.load(args.continue_ckpt))
    
    # Training
    best_loss = torch.inf
    print("Training Start")
    epochs = tqdm(range(args.epochs), leave=False)
    
    for epoch in epochs:
        train_loss = 0
        len_dataloader = min(len(trainloader), len(testloader))
        train_iter = iter(trainloader)
        test_iter = iter(testloader)
        
        model.train()
        for i in range(len_dataloader):
            images, masks = next(train_iter)
            images = images.float().to(device)
            masks = masks.float().to(device)

            optimizer.zero_grad()
            outputs, domain = model(images)
            loss = criterion(outputs, masks.unsqueeze(1))
            domain_label = torch.ones(images.size()[0], 1).float().to(args.device)
            domain_loss = criterion(domain, domain_label)
            
            test_images = next(test_iter).float().to(device)
            _, test_domain = model(test_images)
            test_domain_label = torch.zeros(test_images.size()[0], 1).float().to(args.device)
            domain_test_loss = criterion(test_domain, test_domain_label)
            
            loss = loss + (domain_loss + domain_test_loss) * args.loss_alpha
            loss.backward()
            
            optimizer.step()

            train_loss += loss.item()
        
        scheduler.step()

        train_loss /= len(trainloader)
        current_lr = scheduler.get_lr()[0]

        print(f"Epoch {epoch+1} / train loss: {train_loss}, lr: {current_lr}")
        wandb.log({"train/loss": train_loss})
        wandb.log({"train/lr": current_lr})

        # Inner evaluation step
        if (epoch+1) % args.eval_per_epoch == 0:
            val_loss = 0
            dice = 0.0
            with torch.no_grad():
                model.eval()
                for images, masks in validloader:
                    images = images.float().to(device)
                    masks = masks.float().to(device)

                    outputs, _ = model(images)
                    loss = criterion(outputs, masks.unsqueeze(1))
                    val_loss += loss.item()
                    pred = (torch.sigmoid(outputs) > args.threshold).squeeze(1)
                    dice += DiceScoreSum(pred, masks)

            dice /= (7140 - int(7140 * 0.95)) * 64 # test data num
            val_loss /= len(validloader)
            print(f"Test loss: {val_loss}")
            print(f"Test Dice Score: {dice}")
            wandb.log({"test/loss": val_loss})
            wandb.log({"test/dice": dice})

            if val_loss < best_loss:
                best_loss = val_loss
                directory = args.ckpt_dir + args.model_name + "_" + str(epoch+1)
                print(F"Valid loss decreased. \nSave model on {directory}")

                torch.save(model.state_dict(), directory)

def co_teaching(args):
    trainloader, validloader = get_train_valid_loader(args)

    if args.device == 'cuda':
        if not torch.cuda.is_available():
            print("[Error] Cannot use cuda.")
            return -1

    device = args.device

    wandb.init(
        project="[AI-Competetion] Satellite Image Segmentation",
        name=f"{args.model_name}_{time.strftime('%m%d%H%M', time.localtime())}"
    )
    wandb.config.update(args)
    
    os.makedirs(args.ckpt_dir, exist_ok=True)
    
    model1 = eval('m.' + args.model_name + '().to(device)')
    model2 = eval('m.' + args.model_name + '().to(device)')
    criterion = eval(args.loss+'(reduction=\'none\')')
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=args.lr)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=args.lr)
    total_steps = len(trainloader) * args.epochs
    scheduler1 = CosineAnnealingWarmUpRestarts(optimizer1, T_0=args.lr_period, eta_max=args.max_lr, T_up=args.peak_epoch, gamma=args.lr_gamma)
    scheduler2 = CosineAnnealingWarmUpRestarts(optimizer2, T_0=args.lr_period, eta_max=args.max_lr, T_up=args.peak_epoch, gamma=args.lr_gamma)

    if args.continue_learn:
        model1.load_state_dict(torch.load(args.continue_ckpt + '_1'))
        model2.load_state_dict(torch.load(args.continue_ckpt + '_2'))
    
    # Training
    best_loss1 = torch.inf
    best_loss2 = torch.inf
    print("Training Start")
    epochs = tqdm(range(args.epochs), leave=False)
    
    for epoch in epochs:
        train_loss1 = 0
        train_loss2 = 0
        
        model1.train()
        model2.train()

        num_discard_rate = args.tau * min((epoch * epoch) / args.t_k, 1)

        for images, masks in trainloader:
            images = images.float().to(device)
            masks = masks.float().to(device)

            cur_batch_size = masks.size(0)
            num_discard = num_discard_rate * cur_batch_size

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            outputs1 = model1(images)
            outputs2 = model2(images)
            loss1 = criterion(outputs1, masks.unsqueeze(1)).mean(dim=2).mean(dim=2).squeeze(1)
            loss2 = criterion(outputs2, masks.unsqueeze(1)).mean(dim=2).mean(dim=2).squeeze(1)

            mask_1 = torch.argsort(loss1) >= num_discard
            mask_2 = torch.argsort(loss2) >= num_discard

            loss1 = loss1[mask_2].mean()
            loss2 = loss2[mask_1].mean()

            loss1.backward()
            loss2.backward()
            optimizer1.step()
            optimizer2.step()

            train_loss1 += loss1.item()
            train_loss2 += loss2.item()
        
        scheduler1.step()
        scheduler2.step()

        train_loss1 /= len(trainloader)
        train_loss2 /= len(trainloader)
        current_lr = scheduler1.get_lr()[0]

        print(f"Epoch {epoch+1} / train loss1: {train_loss1}, lr: {current_lr}")
        print(f"Epoch {epoch+1} / train loss2: {train_loss2}, lr: {current_lr}")
        wandb.log({"train/loss1": train_loss1})
        wandb.log({"train/loss2": train_loss2})
        wandb.log({"train/lr": current_lr})

        # Inner evaluation step
        if (epoch+1) % args.eval_per_epoch == 0:
            val_loss1 = 0
            val_loss2 = 0
            dice1 = 0.0
            dice2 = 0.0
            with torch.no_grad():
                model1.eval()
                model2.eval()
                for images, masks in validloader:
                    images = images.float().to(device)
                    masks = masks.float().to(device)

                    outputs1 = model1(images)
                    loss1 = criterion(outputs1, masks.unsqueeze(1)).mean()
                    val_loss1 += loss1.item()
                    pred1 = (torch.sigmoid(outputs1) > args.threshold).squeeze(1)
                    dice1 += DiceScoreSum(pred1, masks)

                    outputs2 = model2(images)
                    loss2 = criterion(outputs2, masks.unsqueeze(1)).mean()
                    val_loss2 += loss2.item()
                    pred2 = (torch.sigmoid(outputs2) > args.threshold).squeeze(1)
                    dice2 += DiceScoreSum(pred2, masks)

            dice1 /= (7140 - int(7140 * 0.95)) * 64 # test data num
            dice2 /= (7140 - int(7140 * 0.95)) * 64 # test data num
            val_loss1 /= len(validloader)
            val_loss2 /= len(validloader)
            print(f"Valid loss1: {val_loss1}")
            print(f"Valid Dice Score1: {dice1}")
            wandb.log({"test/loss1": val_loss1})
            wandb.log({"test/dice1": dice1})

            print(f"Valid loss2: {val_loss2}")
            print(f"Valid Dice Score2: {dice2}")
            wandb.log({"test/loss2": val_loss2})
            wandb.log({"test/dice2": dice2})

            if val_loss1 < best_loss1:
                best_loss1 = val_loss1
                directory = args.ckpt_dir + args.model_name + "_" + str(epoch+1)
                print(F"Valid loss1 decreased. \nSave model on {directory}_1")

                torch.save(model1.state_dict(), directory + '_1')
            
            if val_loss2 < best_loss2:
                best_loss2 = val_loss2
                directory = args.ckpt_dir + args.model_name + "_" + str(epoch+1)
                print(F"Valid loss2 decreased. \nSave model on {directory}_2")

                torch.save(model1.state_dict(), directory + '_1')
                
            if (epoch+1) == args.epochs:
                directory = args.ckpt_dir + args.model_name + "_" + "last"
                print(F"Train finished. Last model will be saved on {directory}_1, {directory}_2")
                
                torch.save(model1.state_dict(), directory + '_1')
                torch.save(model2.state_dict(), directory + '_2')


def main():
    # Argument Parsing (Training Settings)
    args = parseargs()
    if args.mode == 'train':
        train(args)
    elif args.mode == 'submit':
        create_submit_file(args)
    elif args.mode == 'threshold':
        get_threshold(args)
    elif args.mode == 'train_domain':
        train_domain(args)
    elif args.mode == 'co_teaching':
        co_teaching(args)

def get_threshold(args):
    thresholds = list(range(0, 100))
    for i in range(len(thresholds)):
        thresholds[i] /= 100

    device = args.device
    _, validloader = get_train_valid_loader(args)
    dice_scores = [0.0] * len(thresholds)

    model = eval('m.' + args.model_name + '().to(device)')
    model.load_state_dict(torch.load(args.ckpt_dir + args.submit_ckpt))

    with torch.no_grad():
        for images, masks in validloader:
            images = images.float().to(device)
            masks = masks.float().to(device)

            outputs = model(images)

            for i in range(len(thresholds)):
                pred = (torch.sigmoid(outputs) > thresholds[i]).squeeze(1)
                dice_scores[i] += DiceScoreSum(pred, masks)

    best_dice_score = -1
    best_threshold = -1
    for i in range(len(dice_scores)):
        dice_scores[i] /= (7140 - int(7140 * 0.95)) * 64 # test data num
        if dice_scores[i] > best_dice_score:
            best_dice_score = dice_scores[i]
            best_threshold = thresholds[i]
        
        print("threshold :", thresholds[i], "\t\t dice score :", dice_scores[i])
    
    print("best dice score :", best_dice_score)
    print("best threshold :", best_threshold)

def create_submit_file(args):
    # Create Submit File with Best Model
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            print("[Error] Cannot use cuda.")
            return -1
    
    device = args.device

    testloader = get_test_loader(args)
    model = eval('m.' + args.model_name + '().to(device)')
    model.load_state_dict(torch.load(args.ckpt_dir + args.submit_ckpt))
    with torch.no_grad():
        model.eval()
        result = []
        for images in tqdm(testloader):
            images = images.float().to(device)

            outputs = model(images)
            masks = torch.sigmoid(outputs).cpu().numpy()
            masks = np.squeeze(masks, axis=1)
            masks = (masks > args.threshold).astype(np.uint8)

            for i in range(len(images)):
                mask_rle = rle_encode(masks[i])
                if mask_rle == '':
                    result.append(-1)
                else:
                    result.append(mask_rle)
    
    submit = pd.read_csv(args.dataset_path+ './sample_submission.csv')
    submit['mask_rle'] = result
    submit.to_csv(f"{args.model_name}_{time.strftime('%m%d%H%M', time.localtime())}.csv", index=False)

if __name__ == "__main__":
    main()
