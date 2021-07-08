def val_loss(net, dataset, criterion, device='cuda'):
    loss = 0
    cnt = 0
    net.eval()
    for i, (img, label) in enumerate(dataset):
        out = net(img.to(device))
        loss += criterion(out, label.to(device)).item()
        cnt += 1
    net.train()
    loss = loss/cnt
    return loss