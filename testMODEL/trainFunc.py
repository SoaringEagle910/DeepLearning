import torch
def train(model, train_loader, criterion, optimizer, num_epochs=10, device = 'cuda' if torch.cuda.is_available() else 'cpu'):
    model.train()  # 设置模型为训练模式
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print(i,loss.item())
            if i % 100 == 99:  # 每 100 个小批量输出一次损失
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
                running_loss = 0.0
    print("训练完成")

def train_and_save(model, train_loader, criterion, optimizer, num_epochs=10, save_path='./alexnet_mnist.pth'):
    train(model, train_loader, criterion, optimizer, num_epochs)
    torch.save(model.state_dict(), save_path)
    print(f"模型已保存到 {save_path}")