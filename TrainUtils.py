import torch
import numpy as np
import matplotlib.pyplot as plt

class Trainer():

    def __init__(self, training_properties, example, device):
        
        # Loading training properties
        learning_rate = training_properties["learning_rate"]
        weight_decay = training_properties["weight_decay"]
        scheduler_step = training_properties["scheduler_step"]
        scheduler_gamma = training_properties["scheduler_gamma"]
        self.grad_loss = training_properties["grad_loss"]
        
        # Get model and dataset
        self.model = example.model
        n_params = self.model.print_size()
        self.train_loader = example.train_loader #TRAIN LOADER
        self.val_loader = example.val_loader #VALIDATION LOADER
        
        # Set up optimizer and scheduler
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
        
        # Loss Function
        self.loss = torch.nn.MSELoss()
        self.device = device

    def train_epoch(self):
        self.model.train()
        train_mse = 0.0
        running_relative_train_mse = 0.0
        for step, (input_batch, output_batch, vorticity) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            input_batch = input_batch.to(self.device)
            output_batch = output_batch.to(self.device)

            output_pred_batch = self.model(input_batch)

            loss_f = self.loss(output_pred_batch, output_batch) / self.loss(torch.zeros_like(output_batch).to(self.device), output_batch)

            if self.grad_loss:
                pred_dx, pred_dy = torch.gradients(output_pred_batch, edge_order=2)
                output_dx, output_dy = torch.gradients(output_batch, edge_order=2)

                loss_f += self.loss(pred_dx, output_dx)
                loss_f += self.loss(pred_dy, output_dy)

            loss_f.backward()
            self.optimizer.step()
            train_mse = train_mse * step / (step + 1) + loss_f.item() / (step + 1)

        self.scheduler.step()

        return train_mse

    def validate(self):
        
        with torch.no_grad():
            self.model.eval()
            test_relative_l2 = 0.0
            train_relative_l2 = 0.0
            
            for step, (input_batch, output_batch, vorticity) in enumerate(self.val_loader):
                
                input_batch = input_batch.to(self.device)
                output_batch = output_batch.to(self.device)
                output_pred_batch = self.model(input_batch)
                
                loss_f = torch.mean(abs(output_pred_batch - output_batch)) / torch.mean(abs(output_batch)) * 100
                test_relative_l2 += loss_f.item()
            test_relative_l2 /= len(self.val_loader)

            return test_relative_l2

    def log_plots(self):
        """
        Plotting function called when training loop exits.
        Plots the predictions on a batch of the test set and
        uploads them to wandb.
        """
        # Get single test batch
        input_batch, output_batch = next(iter(self.val_loader))
        input_batch = input_batch.to(device)
        output_batch = output_batch.to(device)

        with torch.no_grad():
            pred = self.model(input_batch)
            diffs = output_batch - pred

        # Logging initial conidition channel of inputs as well as outputs and
        # differences between predictions and labels
        input_img = input_batch[0, 2, :, :].detach().cpu().numpy().T
        pred_img = pred[0, 0, :, :].cpu().numpy().T
        label = output_batch[0, 0, :, :].detach().cpu().numpy().T
        diff_img = diffs[0, 0, :, :].cpu().numpy().T

        # Plotting intial condition
        figi, ax = plt.subplots()

        ic = ax.pcolormesh(input_img, cmap="Blues_r")
        ax.set_title("Initial condition")
        figi.colorbar(ic, ax=ax)


        # Plotting predictions
        figp, axes = plt.subplots(1, 2, sharey=True)

        labels = axes[0].pcolormesh(label, cmap="Blues_r")
        axes[0].set_title("Labels")
        figp.colorbar(labels, ax=axes[0])

        fno = axes[1].pcolormesh(pred_img, cmap="Blues_r")
        axes[1].set_title("CNO")
        figp.colorbar(fno, ax=axes[1])

        # Plotting errors
        fige, axes = plt.subplots(1, 2, sharey=True)

        labels = axes[0].pcolormesh(label, cmap="Blues_r")
        axes[0].set_title("Labels")
        fige.colorbar(labels, ax=axes[0])

        error_fno = axes[1].pcolormesh(diff_img, cmap="Blues_r")
        axes[1].set_title("CNO Error")
        fige.colorbar(error_fno, ax=axes[1])

        return figi, figp, fige