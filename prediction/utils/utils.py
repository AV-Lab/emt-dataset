import torch
# def save_model_checkpoint(model, optimizer, history, train_stats=None, additional_info=None, filepath='model_checkpoint.pth'):
#     """
#     Dynamically save model checkpoint excluding device information.
    
#     Args:
#         model: PyTorch model to save
#         optimizer: Model optimizer
#         history: Training history
#         train_stats (dict, optional): Training statistics like mean, std
#         additional_info (dict, optional): Any additional information to save
#         filepath (str): Path where to save the model
#     """
#     # Move model to CPU first
#     model_cpu = model.to('cpu')
    
#     # Get model configuration from its arguments
#     model_config = {
#         key: getattr(model, key) 
#         for key in model_cpu.__init__.__code__.co_varnames
#         if hasattr(model, key) and key != 'device'
#     }
    
#     # Create checkpoint dictionary
#     checkpoint = {
#         'model_state_dict': model_cpu.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'training_history': history,
#         'model_config': model_config,
#         'model_class': model.__class__.__name__
#     }
    
#     # Add training statistics if provided
#     if train_stats is not None:
#         checkpoint['train_stats'] = train_stats
        
#     # Add any additional information
#     if additional_info is not None:
#         checkpoint.update(additional_info)
    
#     # Save the checkpoint
#     torch.save(checkpoint, filepath)
    
#     # Move model back to original device if needed
#     if hasattr(model, 'device'):
#         model = model.to(model.device)

# def load_model_checkpoint(filepath, model_classes, device='cpu'):
#     """
#     Dynamically load saved model checkpoint.
    
#     Args:
#         filepath (str): Path to the saved model
#         model_classes (dict): Dictionary mapping model class names to actual classes
#         device (str): Device to load the model on
        
#     Returns:
#         tuple: (model, optimizer_state, history, train_stats, additional_info)
#     """
#     # Load checkpoint
#     checkpoint = torch.load(filepath, map_location=device)
    
#     # Get model class
#     model_class = model_classes[checkpoint['model_class']]
    
#     # Initialize model with saved configuration
#     model = model_class(**checkpoint['model_config'])
    
#     # Load state dictionary
#     model.load_state_dict(checkpoint['model_state_dict'])
    
#     # Move model to specified device
#     model = model.to(device)
    
#     return (
#         model,
#         checkpoint['optimizer_state_dict'],
#         checkpoint['training_history'],
#         checkpoint.get('train_stats'),
#         {k: v for k, v in checkpoint.items() if k not in [
#             'model_state_dict', 'optimizer_state_dict', 
#             'training_history', 'model_config', 
#             'model_class', 'train_stats'
#         ]}
#     )

# # Usage example:
# # Saving the model
# save_model_checkpoint(
#     model=transformer_model,
#     optimizer=optimizer,
#     history=history,
#     train_stats={'mean': train_mean, 'std': train_std},
#     filepath='transformer_checkpoint.pth'
# )

# # Loading the model
# model_classes = {
#     'Attention_EMT': Attention_EMT  # Add other model classes as needed
# }

# loaded_model, optimizer_state, history, train_stats, additional_info = load_model_checkpoint(
#     filepath='transformer_checkpoint.pth',
#     model_classes=model_classes,
#     device=args.device
# )

# # Reconstruct optimizer if needed
# optimizer = ScheduledOptim(
#     torch.optim.Adam(loaded_model.parameters(), betas=(0.9, 0.98), eps=1e-09),
#     args.lr_mul,
#     args.d_model,
#     args.n_warmup_steps
# )
# optimizer.load_state_dict(optimizer_state)

def generate_square_mask(dim_trg: int, dim_src: int, mask_type: str) -> torch.Tensor:
    """
    Generate a square mask for transformer attention mechanisms.
    
    Args:
        dim_trg (int): Target sequence length.
        dim_src (int): Source sequence length.
        mask_type (str): Type of mask to generate. Can be "src", "tgt", or "memory".
    
    Returns:
        torch.Tensor: A mask tensor with `-inf` values to block specific positions.
    """

    # Initialize a square matrix filled with -inf (default to a fully masked state)
    mask = torch.ones(dim_trg, dim_trg) * float('-inf')

    if mask_type == "src":
        # Source mask (self-attention in the encoder)
        # Creates an upper triangular matrix with -inf above the diagonal
        # This allows each token to attend to itself and previous tokens
        mask = torch.triu(mask, diagonal=1)

    elif mask_type == "tgt":
        # Target mask (self-attention in the decoder)
        # Prevents the decoder from attending to future tokens (causal mask)
        mask = torch.triu(mask, diagonal=1)

    elif mask_type == "memory":
        # Memory mask (cross-attention between encoder and decoder)
        # Controls which encoder outputs the decoder can attend to
        mask = torch.ones(dim_trg, dim_src) * float('-inf')
        mask = torch.triu(mask, diagonal=1)  # Prevents attending to future positions

    return mask


'''
A wrapper class for scheduled optimizer 
source: https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Optim.py
'''
class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, lr_mul, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.lr_mul = lr_mul
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0


    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()


    def zero_grad(self):
        "Zero out the gradients with the inner optimizer"
        self._optimizer.zero_grad()


    def _get_lr_scale(self):
        d_model = self.d_model
        #print(self.n_warmup_steps)
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))


    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_steps += 1
        lr = self.lr_mul * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr