import torch
#returns loss and acc
def val_loss(net, dataset, criterion, device='cuda'):
    loss = 0
    cnt = 0
    net.eval()
    r = 0
    m = 0
    for i, (img, label) in enumerate(dataset):
        out = net(img.to(device))
        if torch.argmax(out).item() == label.item():
            r += 1
        else:
            m += 1
        loss += criterion(out, label.to(device)).item()
        cnt += 1
    net.train()
    loss = loss/cnt
    return loss, r/(r+m)