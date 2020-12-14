import torch
from .utils import AverageMeter
import os 


class Trainer:
    def __init__(self, model, optim, criterion, train_loader, val_loader, val_best_path='./'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optim = optim
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.val_best_path = val_best_path
        self.loss_meter = AverageMeter() 
        self.val_best = float('inf')

        # self.validate(0)
    
    def train(self, epochs, print_feq=50):
        for epoch in range(epochs):
            print("\n")
            self.model.train()
            self.loss_meter.reset()
            for i, (src, trg) in enumerate(self.train_loader):
                self.train_step(src, trg)
                if i%print_feq == 0:
                    print(f'iter {i} loss : {self.loss_meter.avg}')
            print(f'@epoch {epoch} loss : {self.loss_meter.avg}')
            self.validate(epoch)
            
    def train_step(self, x, y):
        self.optim.zero_grad()
        x, y = x.to(self.device), y.to(self.device)
        output = self.model(x, y)

        output = output[1:].view(-1, output.shape[-1])
        y =  y[1:].view(-1)
        loss = self.criterion(output, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)

        self.optim.step()
        self.loss_meter.update(loss.item())

    def validate(self, epoch=0):
        val_meter = AverageMeter()
        self.model.eval()
        with torch.no_grad():
            for _, (src, trg) in enumerate(self.val_loader):
                src, trg = src.to(self.device), trg.to(self.device)
                output = self.model(src, trg, 0) #turn off teacher forcing
                output = output[1:].view(-1, output.shape[-1])
                trg = trg[1:].view(-1)
                loss = self.criterion(output, trg)
                val_meter.update(loss.item())
        print(f'validate loss: {val_meter.avg}')
        if val_meter.avg < self.val_best:
            print('validation best..')
            self.save(self.val_best_path)

    def save(self, save_path):
        torch.save({
            'model_state': self.model.state_dict(),
            }, save_path)
        print(f'model saved at {save_path}')
    
    def load(self, load_path):
        save_dict = torch.load(load_path)
        self.model.load_state_dict(save_dict['model_state'])
        print(f'model loaded from {load_path}')