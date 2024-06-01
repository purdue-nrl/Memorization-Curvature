"""
@author: Anonymized
@copyright: Anonymized
"""

import torch

def inference(net, data_loader, device='cpu', loss=None):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for (data, labels) in data_loader:
            data = data.to(device)
            labels = labels.to(device)
            out = net(data)

            if(loss != None):
                loss_val = loss(out, labels)

            _, pred = torch.max(out, dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size()[0]
        accuracy = float(correct) * 100.0/ float(total)
    
    if(loss != None):
        return correct, total, accuracy, loss_val
    return correct, total, accuracy