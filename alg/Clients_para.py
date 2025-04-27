import copy
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler


class LocalClient:
    def __init__(self, dataset, model_fn, args):
        self.args = args
        self.device = torch.device(args.device)
        self.dataset = dataset
        self.model_fn = model_fn

    def build(self, global_parameters):
        self.net = self.model_fn().to(self.device)
        self.net.load_state_dict(global_parameters)

        self.optimizer = torch.optim.SGD(
            self.net.parameters(),
            lr=self.args.learning_rate,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=30, gamma=0.1
        )

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=2,
        )

    def local_train(self):
        scaler = GradScaler()
        loss_list = []

        for epoch in range(self.args.num_epoch):
            for data, label in self.dataloader:
                data = data.to(self.device)
                label = label.to(self.device)

                with autocast():
                    pred = self.net(data)
                    loss = torch.nn.functional.cross_entropy(pred, label)

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                loss_list.append(loss.item())

            self.scheduler.step()

        avg_loss = sum(loss_list) / len(loss_list)

        # 注意：返回的参数已经detach且clone，避免内存冲突
        cpu_state = {
            k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()
        }

        del self.net
        del self.optimizer
        del self.scheduler
        del self.dataloader
        torch.cuda.empty_cache()

        return cpu_state, avg_loss
