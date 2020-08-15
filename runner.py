#!/usr/bin/env python3.7
import os
import numpy as np
from tqdm import tqdm
# torch imports
import torch       as t
import torch.nn    as tn
import torch.nn    as nn
import torch.optim as to
import torch.optim.lr_scheduler as tolr
import torch.nn.functional as tf
import torchvision.transforms as tt
import torch.utils.data as tud
# ignite imports
import ignite.engine   as ie
import ignite.metrics  as im
import ignite.handlers as ih
# sklearn
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split

class CNN_ONE (nn.Module):
    """
    First design of CNN
    """
    def __init__ (self, idx=100, num_classes=3):
        super (CNN_ONE, self).__init__ ()
        self.name = "ONE"
        self.idx  = idx
        self.input_shape = [1, 114, 80]
        ## features
        self.features = nn.Sequential (
            nn.Conv2d (1, 2, kernel_size=3, stride=3),
            nn.LeakyReLU(),
            nn.Dropout2d(),
            nn.Conv2d (2, 2, kernel_size=3, stride=2),
            nn.LeakyReLU(),
            nn.Dropout2d(),
            nn.Conv2d (2, 2, kernel_size=3, stride=2),
            nn.LeakyReLU(),
            nn.Dropout2d(),
        )
        ## fc
        self.fc = nn.Sequential (
            nn.Linear (80, 27),
            nn.LeakyReLU (),
            nn.Dropout (),
            nn.Linear (27, num_classes),
            # nn.Sigmoid()
        )

    def forward (self, x):
        x = self.features (x)
        x = x.view ([x.size(0), -1])
        return self.fc (x)


class NpyClfDatasets (tud.Dataset):
    """
    Interfaces with Numpy files containing the spectograms
    """
    def __init__ (self, ccsn, ms, chirp, root_dir='./', transform=None):
        """
        Args:
            ccsn (str): CCSN numpy file
            ms(str): Mixed sine numpy file
            chirp (str): chirp numpy file
            transform (callable, optional): Optional transform to be applied

        Convention:
            0 for CCSN
            1 for Mixed sine
            2 for chirp
        """
        self.ccsn  = np.load (os.path.join (root_dir, ccsn))
        self.ms    = np.load (os.path.join (root_dir, ms))
        self.chirp = np.load (os.path.join (root_dir, chirp))
        ##
        self.ncc = self.ccsn.shape[0]
        self.nms = self.ms.shape[0]
        self.nch = self.chirp.shape[0]
        self.n   = self.ncc + self.nms + self.nch
        self.target = np.zeros (self.n, dtype=np.uint8)
        self.target[self.nms:self.nch] = 1
        self.target[self.nch:]         = 2
        self.transform = transform

    def __len__(self):
        return self.n

    def __getitem__ (self, idx):
        """I guess this doesn't support list indexing"""
        if t.is_tensor (idx):
            idx = idx.tolist ()
        ret = dict ()
        if idx < self.ncc:
            ret['payload'] = t.Tensor (self.ccsn[idx].copy()).unsqueeze (0)
            ret['target']  = t.Tensor ([0]).to (t.long)
        elif idx < self.nms + self.ncc:
            ridx = idx - self.ncc
            ret['payload'] = t.Tensor (self.ms[ridx].copy()).unsqueeze (0)
            ret['target']  = t.Tensor ([1]).to (t.long)
        elif idx < self.nch + self.nms + self.ncc:
            ridx = idx - self.ncc - self.nms
            ret['payload'] = t.Tensor (self.chirp[ridx].copy()).unsqueeze (0)
            ret['target']  = t.Tensor ([2]).to (t.long)
        if self.transform:
            ret['payload'] = self.transform (ret['payload'])
        return ret

    def train_test_split (self, test_size=0.3, shuffle=True,random_state=None):
        """returns indices to split train/test"""
        d_i  = np.arange (self.n)
        train_i, test_i = train_test_split (d_i, test_size=test_size, shuffle=shuffle, stratify=self.target, random_state=random_state)
        train_s         = tud.SubsetRandomSampler (train_i)
        test_s          = tud.SubsetRandomSampler (test_i)
        return train_s, test_s

if __name__ == "__main__":
    DSIR = "../"
    MDIR = "cnn/"
    MSS   = "mixed_sine_sxx.npy"
    CHIRP = "chirp_sxx.npy"
    CCSN  = "ccsn_sxx.npy"
    # MSS   = "mixed_sine_sxx_noise.npy"
    # CHIRP = "chirp_sxx_noise.npy"
    # CCSN  = "ccsn_sxx_noise.npy"
    # transforms = tt.Compose ([
        # tt.Normalize (0.0, 1.0)
    # ])
    transforms=None
    DSet = NpyClfDatasets (CCSN, MSS, CHIRP, DSIR, transform=transforms)
    train_l, val_l   = DSet.train_test_split (random_state=24, test_size=0.25)
    t_DataLoader     = tud.DataLoader (DSet, sampler=train_l, batch_size=10, pin_memory=True)
    v_DataLoader     = tud.DataLoader (DSet, sampler=val_l,   batch_size=10, pin_memory=True)
    #########################
    DESC = "Epoch {} - loss {:.2f}"
    PBAR = tqdm (initial=0, leave=False, total=len(t_DataLoader), desc=DESC.format(0, 0))
    CLF  = CNN_ONE(idx=50)
    LFN  = tn.CrossEntropyLoss()
    OPM  = to.Adam(CLF.parameters(), lr=1e-3,)
    VAL_METRICS = {'loss':im.Loss (LFN), 'acc':im.Accuracy()}
    L_TRAIN = []
    L_EVAL  = []
    L_ACC   = []
    #########################
    def train_step(engine, batch):
        CLF.train()
        OPM.zero_grad()
        x, y = batch['payload'], batch['target']
        ypred = CLF (x)
        loss = LFN (ypred, y.squeeze(1))
        loss.backward()
        OPM.step()
        return loss.item()

    def eval_step(engine, batch):
        CLF.eval()
        with t.no_grad():
            x, y = batch['payload'], batch['target']
            y = y.squeeze (1)
            ypred = CLF (x)
            return ypred, y

    TRAINER   = ie.Engine (train_step)
    EVALUATOR = ie.Engine (eval_step)
    for name, metric in VAL_METRICS.items():
        metric.attach (EVALUATOR, name)
    #########################
    TO_CHECKP  = {
        "trainer":TRAINER,
        "evaluator":EVALUATOR,
        "model":CLF,
        "optimizer":OPM,
    }
    tckp = ih.Checkpoint (
        to_save = TO_CHECKP,
        save_handler = ih.DiskSaver (MDIR, require_empty=False),
        n_saved=10,
    )
    TRAINER.add_event_handler (ie.Events.EPOCH_COMPLETED (every=50), tckp)
    ###########
    ## resume logic
    if False:
        RFROM = "/home/shining/mega/machine_learnings/zebelblast/checkpoint_1357200.pt"
        tqdm.write ("Resuming from {}".format(RFROM))
        chkp = t.load (RFROM)
        ih.Checkpoint.load_objects (to_load=TO_CHECKP, checkpoint=chkp)
    ###########
    @TRAINER.on (ie.Events.ITERATION_COMPLETED(every=10))
    def log_training_loss (engine):
        PBAR.desc = DESC.format (engine.state.epoch, engine.state.output)
        PBAR.update (10)

    @TRAINER.on (ie.Events.EPOCH_COMPLETED)
    def log_training_results (TRAINER):
        PBAR.refresh()
        EVALUATOR.run (t_DataLoader)
        metrics = EVALUATOR.state.metrics
        tqdm.write ("Training   :: Epoch {} Loss {:.2f}".format (TRAINER.state.epoch, np.log10(metrics['loss'])))
        L_TRAIN.append (metrics['loss'])
        PBAR.n = PBAR.last_print_n = 0

    @TRAINER.on (ie.Events.EPOCH_COMPLETED)
    def log_validation_results (TRAINER):
        EVALUATOR.run (v_DataLoader)
        metrics = EVALUATOR.state.metrics
        tqdm.write ("Validation :: Epoch {} Loss {:.2f} Acc {:.2f}".format (TRAINER.state.epoch, np.log10(metrics['loss']), 100*metrics['acc']))
        L_EVAL.append (metrics['loss'])
        L_ACC.append (metrics['acc'])
        PBAR.n = PBAR.last_print_n = 0

    def loss_score (engine):
        return -engine.state.metrics['loss']
    early_stopper = ih.EarlyStopping (patience=10,score_function=loss_score,trainer=TRAINER)
    EVALUATOR.add_event_handler (ie.Events.COMPLETED, early_stopper)
    #########################
    try:
        TRAINER.run (t_DataLoader, max_epochs=100)
        PBAR.close ()
    except KeyboardInterrupt:
        print ("Received keyboard interrupt")
    ######
    with open (os.path.join (MDIR, "losses1k.pkl"), 'wb') as lf:
        import pickle as pkl
        pkl.dump ([L_TRAIN, L_EVAL, L_ACC], lf)
