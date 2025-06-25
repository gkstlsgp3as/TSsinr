import os
import numpy as np
import torch
import setup
import losses
import models
import datasets
import utils
import torch.nn as nn

class Trainer():

    def __init__(self, model, train_loader, params):

        self.params = params

        # define loaders:
        self.train_loader = train_loader

        # define model:
        self.model = model
        
        self.device = params['device']
        self.model.to(self.device)

        # define important objects:
        self.compute_loss = losses.get_loss_function(params)
        self.encode_location = self.train_loader.dataset.enc.encode

        # define optimization objects:
        self.optimizer = torch.optim.Adam(self.model.parameters(), params['lr'])
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=params['lr_decay'])
        
        # 로그 파일 준비
        log_path = os.path.join(self.params['save_path'], 'train_log.txt')
        self.log_file = open(log_path, 'a')

    def train_one_epoch(self):

        self.model.train()
        # initialise run stats
        running_loss = 0.0
        samples_processed = 0
        steps_trained = 0
        for _, batch in enumerate(self.train_loader):
            # reset gradients:
            self.optimizer.zero_grad()
            # compute loss:
            batch_loss = self.compute_loss(batch, self.model, self.params, self.encode_location)
            # backwards pass:
            batch_loss.backward()
            # update parameters:
            self.optimizer.step()
            # track and report:
            running_loss += float(batch_loss.item())
            steps_trained += 1
            samples_processed += batch[0].shape[0]
            if steps_trained % self.params['log_frequency'] == 0:
                avg_loss = running_loss / self.params["log_frequency"]
                message = f'[{samples_processed}/{len(self.train_loader.dataset)}] loss: {np.around(avg_loss, 4)}'
                self.log(message)
                running_loss = 0.0
        # update learning rate according to schedule:
        self.lr_scheduler.step()
        
    def train_one_epoch_latent(self):
        self.model.train()

        running_loss = 0.0
        samples_processed = 0
        steps_trained = 0

        latent_cache = {}  # (lon, lat) -> latent vector

        for _, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            loc_feat, loc, obs, time_feat, key = batch  # unpack
            
            loc_feat = loc_feat.to(self.device)
            obs = obs.to(self.device)
            time_feat = time_feat.to(self.device)

            # prepare previous latent
            prev_latents = []
            for k in key:
                if k in latent_cache:
                    prev_latents.append(latent_cache[k])
                else:
                    prev_latents.append(torch.zeros(self.model.latent_dim).to(loc_feat.device))

            prev_latents = torch.stack(prev_latents, dim=0)  # shape: (B, latent_dim)

            # forward pass
            pred, latents = self.model(loc_feat, prev_latents)

            # compute loss
            loss_fn = self.params.get("loss_fn", nn.MSELoss())
            batch_loss = loss_fn(pred, obs)

            batch_loss.backward()
            self.optimizer.step()

            # update cache
            for i, k in enumerate(key):
                latent_cache[k] = latents[i].detach()

            # log
            running_loss += float(batch_loss.item())
            steps_trained += 1
            samples_processed += loc_feat.shape[0]
            if steps_trained % self.params['log_frequency'] == 0:
                print(f'[{samples_processed}/{len(self.train_loader.dataset)}] loss: {np.around(running_loss / self.params["log_frequency"], 4)}')
                running_loss = 0.0

        self.lr_scheduler.step()

    def save_model(self, epoch=None):
        if epoch is not None:
            filename = f"model_epoch{epoch}.pt"
        else:
            filename = "model.pt"
        save_path = os.path.join(self.params['save_path'], filename)
        op_state = {'state_dict': self.model.state_dict(), 'params': self.params}
        torch.save(op_state, save_path)
        
    def log(self, message):
        print(message)
        self.log_file.write(message + '\n')
        self.log_file.flush()

def launch_training_run(ovr):
    # setup:
    params = setup.get_default_params_train(ovr)
    params['save_path'] = os.path.join(params['save_base'], params['experiment_name'])
    if params['timestamp']:
        params['save_path'] = params['save_path'] + '_' + utils.get_time_stamp()
    os.makedirs(params['save_path'], exist_ok=False)

    # data:
    train_dataset = datasets.get_train_data(params)
    params['input_dim'] = train_dataset.input_dim
    #params['num_classes'] = train_dataset.num_classes
    #params['class_to_taxa'] = train_dataset.class_to_taxa
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=params['batch_size'],
        shuffle=True,
        num_workers=4)
    print("dataset loaded..")
    # model:
    model = models.get_model(params)
    model = model.to(params['device'])
    print("model loaded..")
    # train:
    print("start training..")
    trainer = Trainer(model, train_loader, params)
    
    if params['latent']:
        for epoch in range(params['num_epochs']):
            print(f'epoch {epoch+1}')
            trainer.train_one_epoch_latent()
            if (epoch + 1) % 5 == 0:
                trainer.save_model(epoch + 1)
    else:
        for epoch in range(params['num_epochs']):
            print(f'epoch {epoch+1}')
            trainer.train_one_epoch()
            if (epoch + 1) % 5 == 0:
                trainer.save_model(epoch + 1)
    trainer.save_model(epoch + 1)
    trainer.log_file.close()
