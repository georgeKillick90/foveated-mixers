import torch
import torch.nn.functional as F

def train(
    model,
    optimizer,
    scheduler,
    scaler,
    train_loader,
    valid_loader,
    epochs,
    device='cuda'
    ):
        
    model.to(device)
    
    def print_summary(tag, metrics):
        summary = "\nAverage {tag} Loss: {loss:.2f} - Average {tag} Accuracy {acc:.2f}"
        print(summary.format(tag=tag, loss=train_metrics['loss'], acc=train_metrics['acc']))
        
    
    print("### Training ###")
    
    for epoch in range(epochs):
        
        # just a progress bar 
        prog = int(30 * epoch / epochs)
        progress_bar = "[" + "#" * prog + "-" * (30-prog) + "]"
        print(("\nEpoch {}: "+ progress_bar + "\n").format(epoch))
        
        # trains an epoch and prints some summary statistics
        
        train_metrics = train_epoch(model, optimizer, scheduler, scaler, train_loader)
        
        print_summary("Train", train_metrics)
        
        # runs a validation epoch and prints some summary statistics
                
        print("\n### Validation and Checkpointing ###")
        
        val_metrics = test(model, valid_loader)
        
        print_summary("Validation", val_metrics)
        
        # extra loop logic goes here : TODO
            
def train_epoch(model, optimizer, scheduler, scaler, train_loader, device='cuda'):

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    model.train()
    
    # with autocast for mixed precision
    with torch.autocast(device_type=device, dtype=torch.float16):
        
        for i, (images, labels) in enumerate (train_loader):
            
            # forward
            
            images, labels = images.to(device), labels.to(device)
            
            predictions = model(images)
            
            loss = F.cross_entropy(predictions, labels, label_smoothing=0.1)
            
            # backwards with mixed precision
            
            scaler.scale(loss).backward()
            
            # optimizer and scheduler step + grad zeroing

            scaler.step(optimizer)

            scaler.update()

            optimizer.zero_grad(set_to_none=True)

            scheduler.step()
            
            # check loss is not nan
            
            assert not torch.isnan(loss), "loss is nan"
            
            # compute accuracy for batch
            
            acc = accuracy(predictions, labels)
            
            # update the metric meters to track statistics over epoch
            
            loss_meter.update(loss.item(), n=images.shape[0])
            acc_meter.update(acc, n=images.shape[0])
            
            # prints current batch loss and accuracy every 5 batches
            
            if i%5 == 0:
                msg = "Batch: {batch} - Loss: {loss:.2f} --- Accuracy: {accuracy:.2f}"
                print(msg.format(batch=i, loss=loss_meter.val, accuracy=acc_meter.val))
            
    return {'loss': loss_meter.avg, 'acc': acc_meter.avg}

def test(model, valid_loader, device='cuda'):
    
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    model.eval()
    
    with torch.no_grad():
        
        with torch.autocast(device_type=device, dtype=torch.float16):
        
            for i, (images, labels) in enumerate(valid_loader):
                
                images, labels = images.to(device), labels.to(device)
                
                predictions = model(images)
                
                loss = F.cross_entropy(predictions, labels)
                
                assert not torch.isnan(loss), "loss is nan"
                
                acc = accuracy(predictions, labels)
                
                loss_meter.update(loss.item(), n=images.shape[0])
                acc_meter.update(acc, n=images.shape[0])
    
    return {"loss": loss_meter.avg, 'acc': acc_meter.avg}

class AverageMeter(object):
    """computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        

def accuracy(predictions, targets):
    predictions = torch.argmax(predictions, dim=-1)
    acc = torch.eq(predictions, targets).float().mean()
    return acc * 100