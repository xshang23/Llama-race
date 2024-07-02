import torch
from torch import nn

# def fair_loss(logits, labels, tokenizer, lambda_val=0.1,loss_scale=True):
#     class_names = ["Asian", "Black", "Hispanic", "White"]
#     class_tokens = [tokenizer.encode(class_name, add_special_tokens=False)[0] for class_name in class_names]
#     class_token_lengths = [len(tokenizer.encode(class_name, add_special_tokens=False)) for class_name in class_names]
#     logits_indices = []
#     for i in range(len(labels)):
#         logits_indices.append(class_token_lengths[labels[i].item()]*(-1))
#         # print(labels[i].item())
#     # logits_indices = [class_token_lengths[class_index.item()] for class_index in labels]*(-1)
#     print('debugging: logits_indices, labels')
#     print(len(logits_indices), len(labels))
#     last_token_logits = logits[:, logits_indices, :]  # Shape: [batch_size, vocabulary_size]


#     # extract the logits for each class token
#     class_logits = last_token_logits[:, class_tokens]  # Shape: [batch_size, num_classes]

#     labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=4)
#     # print(labels_one_hot)

#     # Initialize the BCEWithLogitsLoss function
#     bce_loss_fn = nn.BCEWithLogitsLoss(reduction='none')

#     # Ensure class_logits and labels_one_hot are of correct shape and type
#     # class_logits: [batch_size, num_classes]
#     # labels_one_hot: [batch_size, num_classes], make sure it's a float tensor

#     # Apply sigmoid and calculate loss for each class individually
#     losses = bce_loss_fn(class_logits, labels_one_hot.float())
#     class_losses = losses.mean(dim=0)

#     # Calculate the distance between the minorty and majority class
#     majority_class_loss = class_losses[-1]
#     total_loss = torch.tensor(0.0, requires_grad=True)
#     for i in range(len(class_losses)-1):
#         minority_class_loss = class_losses[i]
#         distance = (minority_class_loss - majority_class_loss)**2
#         total_loss = total_loss + distance
#     if not loss_scale:
#         return lambda_val*total_loss
#     return lambda_val*total_loss*10e2


def fair_loss(logits, labels, tokenizer, lambda_val=0.1, loss_scale=True):
    class_names = ["Asian", "Black", "Hispanic", "White"]
    # Tokenize all class names and store the first token (assumed to be the main representative token)
    class_tokens = [tokenizer.encode(class_name, add_special_tokens=False)[0] for class_name in class_names]
    # Store lengths of each tokenized class name
    class_token_lengths = [len(tokenizer.encode(class_name, add_special_tokens=False)) for class_name in class_names]

    class_logits = []
    # Iterate over each example in the batch to handle variable token lengths at sequence end
    for i in range(logits.size(0)):
        label_idx = labels[i].item()  # Get label as an integer index
        num_tokens = class_token_lengths[label_idx]  # Get the number of tokens for the class name
        # Extract the logits for the last 'num_tokens' positions (mean for simplicity)
        selected_logits = logits[i, -num_tokens, :]
        class_logits.append(selected_logits)

    # Stack all selected logits into a batch tensor
    class_logits = torch.stack(class_logits)
    class_logits = class_logits[:, class_tokens]
    labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=4)

    # Initialize the BCEWithLogitsLoss function
    bce_loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    losses = bce_loss_fn(class_logits, labels_one_hot.float())
    class_losses = losses.mean(dim=0)

    # Calculate the distance between the minority and majority class losses
    majority_class_loss = class_losses[-1]
    total_loss = torch.tensor(0.0, requires_grad=True)
    for i in range(len(class_losses)-1):
        minority_class_loss = class_losses[i]
        distance = (minority_class_loss - majority_class_loss)**2
        total_loss = total_loss + distance

    if not loss_scale:
        return lambda_val * total_loss
    return lambda_val * total_loss * 10e2
 


