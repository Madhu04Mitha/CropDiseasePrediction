model = models.vgg16(pretrained=True) 
 
 
#Modelling 
 
   
 class CNN(nn.Module): 
    def __init__(self, K): 
        super(CNN, self).__init__() 
        self.conv_layers = nn.Sequential( 
            # conv1 
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1), 
            nn.ReLU(), 
            nn.BatchNorm2d(32), 
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1), 
            nn.ReLU(), 
            nn.BatchNorm2d(32), 
            nn.MaxPool2d(2), 
            # conv2 
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1), 
            nn.ReLU(), 
            nn.BatchNorm2d(64), 
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1), 
            nn.ReLU(), 
            nn.BatchNorm2d(64), 
            nn.MaxPool2d(2), 
            # conv3 
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1), 
            nn.ReLU(), 
            nn.BatchNorm2d(128), 
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1), 
            nn.ReLU(), 
            nn.BatchNorm2d(128), 
            nn.MaxPool2d(2), 
            # conv4 
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1), 
            nn.ReLU(),                     
 
            nn.BatchNorm2d(256),                          16 
 
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1), 
            nn.ReLU(), 
            nn.BatchNorm2d(256), 
            nn.MaxPool2d(2), 
        ) 
 
        self.dense_layers = nn.Sequential( 
            nn.Dropout(0.4), 
            nn.Linear(50176, 1024), 
            nn.ReLU(), 
            nn.Dropout(0.4), 
            nn.Linear(1024, K), 
        ) 
 
    def forward(self, X): 
        out = self.conv_layers(X) 
 
        # Flatten 
        out = out.view(-1, 50176) 
 
        # Fully connected 
        out = self.dense_layers(out) 
 
        return out 
    
 
#Batch Gradient Descent 
 
def batch_gd(model, criterion, train_loader, test_laoder, epochs): 
    train_losses = np.zeros(epochs) 
    validation_losses = [] 
    for e in range(epochs): 
        train_loss = [] 
        for inputs, targets in train_loader: 
            inputs, targets = inputs.to(device), targets.to(device) 
 
            # zero the parameter gradients 
            optimizer.zero_grad() 
 
            # forward + backward + optimize 
            outputs = model(inputs) 
            loss = criterion(outputs, targets) 
            loss.backward()                                            17 
 
            optimizer.step() 
 
            train_loss.append(loss.item()) 
 
        # calculate validation loss 
        validation_loss = [] 
        for inputs, targets in test_loader: 
            inputs, targets = inputs.to(device), targets.to(device) 
 
            outputs = model(inputs) 
            loss = criterion(outputs, targets) 
 
            validation_loss.append(loss.item()) 
 
        train_loss = np.mean(train_loss) 
        validation_loss = np.mean(validation_loss) 
 
        train_losses[e] = train_loss 
        validation_losses.append(validation_loss) 
        validation_losses[e] = validation_loss 
 
        print(f'Epoch {e+1}, Train Loss: {train_loss:.4f}, Validation Loss: {validation_loss:.4f}') 
 
    return train_losses, validation_losses