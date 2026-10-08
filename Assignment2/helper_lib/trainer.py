import torch


def train_model(model, data_loader, criterion, optimizer, device="cpu", epochs=10):
    model.to(device)
    history = []
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
        row = {"epoch": epoch + 1, "train_loss": total_loss / total,
               "train_accuracy": correct / total}
        history.append(row)
        print(f"Epoch {epoch + 1}/{epochs}: loss={row['train_loss']:.4f}, "
              f"train accuracy={row['train_accuracy']:.2%}")
    model.training_history = history
    return model


def evaluate_model(model, data_loader, criterion, device="cpu"):
    model.to(device)
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return {"test_loss": total_loss / total, "test_accuracy": correct / total,
            "test_samples": total}
