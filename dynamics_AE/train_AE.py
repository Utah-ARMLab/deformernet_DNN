import numpy as np
from tqdm import tqdm
import argparse
import os
from os.path import join, exists

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

from models import *
import utils as cu 


def train(encoder, decoder, trans, optimizer, train_loader, epoch, device):
    trans.train()

    stats = cu.Stats()
    pbar = tqdm(total=len(train_loader.dataset))
    parameters = list(encoder.parameters()) + list(decoder.parameters()) + list(trans.parameters())
    for batch in train_loader:
        obs, obs_pos, actions = [b.to(device) for b in batch]
        z, saved_points = encoder(obs, decode=True)
        z_pos= encoder(obs_pos)
        z_pred = trans(z, actions)
        obs_recon = decoder(saved_points)

        recon_loss = F.mse_loss(obs_recon, obs)
        trans_loss = F.mse_loss(z_pred, z_pos)
        loss = recon_loss + trans_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(parameters, 1)
        optimizer.step()

        stats.add('train_loss', loss.item())
        stats.add('recon_loss', recon_loss.item())
        stats.add('trans_loss', trans_loss.item())
        avg_loss = np.mean(stats['train_loss'][-50:])
        avg_recon_loss = np.mean(stats['recon_loss'][-50:])
        avg_trans_loss = np.mean(stats['trans_loss'][-50:])

        pbar.set_description(f'Epoch {epoch}, Train Loss {avg_loss:.4f}, '
                             f'Recon Loss {avg_recon_loss:.4f}, Trans Loss {avg_trans_loss:.4f}')
        pbar.update(obs.shape[0])
    pbar.close()


def test(encoder, decoder, trans, test_loader, epoch, device):
    trans.eval()

    test_loss, test_recon_loss, test_trans_loss = 0, 0, 0
    for batch in test_loader:
        with torch.no_grad():
            obs, obs_pos, actions = [b.to(device) for b in batch]
            z, saved_points = encoder(obs, decode=True)
            z_pos= encoder(obs_pos)
            z_pred = trans(z, actions)
            obs_recon = decoder(saved_points)

            recon_loss = F.mse_loss(obs_recon, obs)
            trans_loss = F.mse_loss(z_pred, z_pos)
            loss = recon_loss + trans_loss

            test_loss += loss * obs.shape[0]
            test_recon_loss += recon_loss * obs.shape[0]
            test_trans_loss += trans_loss * obs.shape[0]
    test_loss /= len(test_loader.dataset)
    test_recon_loss /= len(test_loader.dataset)
    test_trans_loss /= len(test_loader.dataset)
    print(f'Epoch {epoch}, Test Loss: {test_loss:.4f}, Recon Loss: {test_recon_loss:.4f}, Trans Loss: {test_trans_loss:.4f}')
    return test_loss.item()

def main():   
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # folder_name = join('out', args.name)
    folder_name = "/home/baothach/med_robot_course/learned_dynamics/autoencoder/run1"
    
    device = torch.device('cuda')
    
    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    action_dim = 3
    z_dim = 256
    # trans = TransitionSimple(z_dim, action_dim, trans_type=args.trans_type, hidden_size=64).to(device)
    trans = TransitionSimple(z_dim, action_dim, trans_type='mlp', hidden_size=256).to(device)


    parameters = list(encoder.parameters()) + list(decoder.parameters()) + list(trans.parameters())

    optimizer = optim.Adam(parameters, lr=args.lr)
    train_loader, test_loader = cu.get_dataloaders(args.root, args.batch_size, train_len=9500, test_len=500)

    best_test_loss = float('inf')
    for epoch in range(args.epochs):
        train(encoder, decoder, trans, optimizer, train_loader, epoch, device)
        test_loss = test(encoder, decoder, trans, test_loader, epoch, device)

        if epoch % args.log_interval == 0:
            if test_loss <= best_test_loss:
                best_test_loss = test_loss

                checkpoint = {
                    'encoder': encoder,
                    'trans': trans,
                    'decoder': decoder,
                    'optimizer': optimizer,
                }
                torch.save(checkpoint, join(folder_name, 'checkpoint'))
                print('Saved models with loss', best_test_loss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Dataset Parameters
    parser.add_argument('--root', type=str, default='/home/baothach/shape_servo_data/RL_shapeservo/box/processed_data/', help='path to dataset (default: data/pointmass)')

    # Architecture Parameters

    parser.add_argument('--trans_type', type=str, default='linear',
                        help='linear | mlp | reparam_w | reparam_w_ortho_gs | reparam_w_ortho_cont | reparam_w_tanh (default: linear)')

    # Learning Parameters
    parser.add_argument('--lr', type=float, default=1e-3, help='base learning rate for batch size 128 (default: 1e-3)')
    parser.add_argument('--epochs', type=int, default=100, help='default: 100')
    parser.add_argument('--log_interval', type=int, default=1, help='default: 1')
    parser.add_argument('--batch_size', type=int, default=32, help='default 32')

    # Other
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--name', type=str, default='autoencoder')
    args = parser.parse_args()

    main()    