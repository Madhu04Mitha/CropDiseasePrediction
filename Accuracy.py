batch_size = 32 
train_loader = torch.utils.data.DataLoader( 
    dataset, batch_size=batch_size, sampler=train_sampler 
) 
test_loader = torch.utils.data.DataLoader( 
    dataset, batch_size=batch_size, sampler=test_sampler 
) 
validation_loader = torch.utils.data.DataLoader( 
    dataset, batch_size=batch_size, sampler=validation_sampler 
) 
 
train_losses, validation_losses = batch_gd( 
    model, criterion, train_loader, validation_loader, 2 
)                                                                           
 
 
#Calculating Accuracy 
def accuracy(loader): 
    n_correct = 0 
    n_total = 0 
 
    with torch.no_grad(): 
        for inputs, targets in loader: 
            # Make predictions using your model 
            outputs = model(inputs) 
            # Get the index of the predicted class 
            _, predicted = torch.max(outputs, 1) 
            # Count the number of correct predictions 
            n_correct += (predicted == targets).sum().item() 
            # Count the total number of samples 
            n_total += targets.shape[0] 
 
        # Check if n_total is zero before calculating accuracy 
        if n_total == 0: 
            acc = 0 
        else: 
            acc = n_correct / n_total 
    return acc 