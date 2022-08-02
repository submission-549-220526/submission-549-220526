import torch
import torch.nn as nn
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import time
import os

class CustomCheckPointCallback(ModelCheckpoint):
    def on_train_start(self, trainer, pl_module):
        rank_zero_info("\nModel Version: " + pl_module.logger.version)
    def on_train_end(self, trainer, pl_module):
        super().on_train_end(trainer, pl_module)

        rank_zero_info("\nModel Version: " + pl_module.logger.version)
        rank_zero_info("Model saved under: " + self.last_model_path)

class PodTrainer():
    def fit(self, net, dm):
        print('Training begin.')
        dm.setup()
        trainloader = dm.train_dataloader()
        snapshots = None
        for i_batch, data_batched in enumerate(trainloader):
            q = data_batched['q']
            if snapshots is None:
                snapshots = q
            else:
                snapshots = torch.cat([snapshots, q], 0)
        snapshots = snapshots.view(snapshots.size(0), -1)
        snapshots = torch.transpose(snapshots, 0, 1) # m (vector field values) by n (time-steps), m >= n
        assert(snapshots.size(0)>=snapshots.size(1))

        U, S, V = torch.pca_lowrank(snapshots, net.lbllength)

        encoder_matrix = torch.inverse(torch.matmul(U.transpose(1, 0), U)).matmul(U.transpose(1, 0))
        decoder_matrix = U

        net.encoder.linear_layer.weight = nn.Parameter(encoder_matrix.clone())
        net.decoder.linear_layer.weight = nn.Parameter(decoder_matrix.clone())
        print('Training finished.')

        output_path = './outputs'
        time_string = time.strftime("%Y%m%d-%H%M%S")
        weightdir = output_path + '/weights/' + time_string
        checkpoint_callback = CustomCheckPointCallback(verbose=True, dirpath=weightdir, filename='{epoch}-{step}')

        trainer = Trainer(default_root_dir=output_path, max_epochs=0, logger=False)
        trainer.fit(net, dm)
        ckpt_path = os.path.join(weightdir, 'weights.ckpt')
        trainer.save_checkpoint(ckpt_path)
        print('saved weight path: ', ckpt_path)
