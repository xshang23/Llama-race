import torch
from torch import nn

def fair_loss(logits, labels, tokenizer, lambda_val=0.1):
    class_names = ["Asian", "Black", "Hispanic", "White"]
    class_tokens = [tokenizer.encode(class_name, add_special_tokens=False)[-1] for class_name in class_names]
    last_token_logits = logits[:, -1, :]  # Shape: [batch_size, vocabulary_size]

    # Now, extract the logits for each class token
    # Assuming you've obtained class_tokens_ids from the previous step
    class_logits = last_token_logits[:, class_tokens]  # Shape: [batch_size, num_classes]

    labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=4)
    # print(labels_one_hot)

    # Initialize the BCEWithLogitsLoss function
    bce_loss_fn = nn.BCEWithLogitsLoss(reduction='none')

    # Ensure class_logits and labels_one_hot are of correct shape and type
    # class_logits: [batch_size, num_classes]
    # labels_one_hot: [batch_size, num_classes], make sure it's a float tensor

    # Apply sigmoid and calculate loss for each class individually
    losses = bce_loss_fn(class_logits, labels_one_hot.float())
    class_losses = losses.mean(dim=0)

    # Calculate the distance between the minorty and majority class
    majority_class_loss = class_losses[-1]
    total_loss = torch.tensor(0.0, requires_grad=True)
    for i in range(len(class_losses)-1):
        minority_class_loss = class_losses[i]
        distance = (minority_class_loss - majority_class_loss)**2
        total_loss = total_loss + distance
    
    return lambda_val*total_loss


    

