import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import datetime
from collections import defaultdict

# Define the CNN architecture
class CIFAR10CNN(nn.Module):
    def __init__(self):
        super(CIFAR10CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) #ABLATABLE_COMPONENT
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 32x16x16
        x = self.pool(F.relu(self.conv2(x)))  # 64x8x8
        x = self.pool(F.relu(self.conv3(x)))  # 128x4x4
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Training function with epoch-wise validation and TensorBoard logging
def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=20):
    # Initialize TensorBoard writer
    current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    writer = SummaryWriter(f'runs/cifar10_experiment_{current_time}')
    
    best_acc = 0.0
    
    # Log model graph
    example_images, _ = next(iter(train_loader))
    writer.add_graph(model, example_images.to(device))

    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            epoch_total += labels.size(0)
            epoch_correct += (predicted == labels).sum().item()

            # Print batch progress
            if i % 100 == 99:
                print(f'Epoch {epoch + 1}, Batch {i + 1}: training loss: {loss.item():.3f}')

        # Calculate and log training metrics for the epoch
        avg_train_loss = epoch_loss / len(train_loader)
        train_accuracy = 100 * epoch_correct / epoch_total
        
        writer.add_scalar('Training/Loss', avg_train_loss, epoch)
        writer.add_scalar('Training/Accuracy', train_accuracy, epoch)

        print(f'Epoch {epoch + 1} Training - Avg Loss: {avg_train_loss:.3f}, '
              f'Accuracy: {train_accuracy:.2f}%')

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data in val_loader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        # Calculate and log validation metrics
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        
        writer.add_scalar('Validation/Loss', avg_val_loss, epoch)
        writer.add_scalar('Validation/Accuracy', val_accuracy, epoch)
        
        # Log learning rate
        writer.add_scalar('Training/Learning_Rate', 
                         optimizer.param_groups[0]['lr'], 
                         epoch)
        
        print(f'Epoch {epoch + 1} Validation - Avg Loss: {avg_val_loss:.3f}, '
              f'Accuracy: {val_accuracy:.2f}%')

        # Calculate and log the gap between training and validation metrics
        writer.add_scalar('Metrics/Train_Val_Loss_Gap', 
                         abs(avg_train_loss - avg_val_loss), epoch)
        writer.add_scalar('Metrics/Train_Val_Accuracy_Gap', 
                         abs(train_accuracy - val_accuracy), epoch)

        # Save best model
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, 'best_model.pth')
            print(f'New best model saved with accuracy: {best_acc:.2f}%')

    writer.close()

# Data loading and training setup
def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=train_transform)
    valset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=val_transform)

    trainloader = DataLoader(trainset, batch_size=128,
                           shuffle=True, num_workers=2)
    valloader = DataLoader(valset, batch_size=128,
                          shuffle=False, num_workers=2)

    # Initialize model, loss function, and optimizer
    model = CIFAR10CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, trainloader, valloader, criterion, optimizer, device)

if __name__ == '__main__':
    main()
    