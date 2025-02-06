
from tqdm import tqdm
import torch
import torch.nn as nn
import matplotlib as plt
import numpy as np
import scipy.spatial
import scipy.io
import matplotlib.pyplot as plt
# from utils.utils import generate_square_mask
from evaluation.distance_metrics import calculate_ade,calculate_fde

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


def calculate_metrics(pred, target, obs_last_pos, normalized, mean, std, device):
    """
    Calculate ADE and FDE for predictions
    Args:
        pred: predicted velocities [batch, seq_len, 2]
        target: target velocities [batch, seq_len, 2]
        obs_last_pos: last observed position [batch, 1, 2]
        normalized: whether predictions are normalized
    """
    if normalized:
        # Denormalize
        pred = pred * std[2:].to(device) + mean[2:].to(device)
        target = target * std[2:].to(device) + mean[2:].to(device)
    
    # Convert velocities to absolute positions through cumsum
    pred_pos = pred.cpu().numpy().cumsum(1) + obs_last_pos.cpu().numpy()
    target_pos = target.cpu().numpy().cumsum(1) + obs_last_pos.cpu().numpy()
    
    # print(pred_pos, target_pos)dddd

    # Calculate metrics change to list from numpy array

    ade = calculate_ade(pred_pos, target_pos.tolist())
    fde = calculate_fde(pred_pos, target_pos.tolist())
    # mad, fad, _ = distance_metrics(target_pos, pred_pos)
    
    return ade, fde

def train_attn(args, train_dl, test_dl, model=None, optim=None, epochs=100, 
               mean=torch.tensor([0.0, 0.0, 0.0, 0.0]), 
               std=torch.tensor([1.0, 1.0, 1.0, 1.0])):
    
    print('Training Settings:')
    print(f"Train batch size: {args.batch_size}")
    print(f"Epochs: {epochs}")

    mean = mean.to(args.device)
    std = std.to(args.device)
    criterion = nn.MSELoss()
    
    # Initialize tracking
    train_losses, test_losses = [], []
    train_ades, test_ades = [], []
    train_fdes, test_fdes = [], []

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_ade = 0
        epoch_fde = 0
        model.train()

        load_train = tqdm(train_dl, desc=f"Epoch: {epoch+1}/{epochs}")

        for id_b, batch in enumerate(load_train):
            obs_tensor, target_tensor = batch
            batch_size, enc_seq_len, feat_dim = obs_tensor.shape
            dec_seq_len = target_tensor.shape[1]
            
            obs_tensor = obs_tensor.to(args.device)
            target_tensor = target_tensor.to(args.device)

            input = (obs_tensor[:,1:,2:4] - mean[2:])/std[2:]
            updated_enq_length = input.shape[1]
            target = (target_tensor[:,:,2:4] - mean[2:])/std[2:]

            tgt = torch.zeros_like(target).to(args.device)  
            tgt[:, 1:, :] = target[:, :-1, :] 

            tgt_mask = generate_square_mask(dim_trg=dec_seq_len, 
                                          dim_src=updated_enq_length, 
                                          mask_type="tgt").to(args.device)
            
            memory_mask = generate_square_mask(
                dim_trg=dec_seq_len,  # = 10
                dim_src=updated_enq_length,   # = 9
                mask_type="memory"
            ).to(args.device)


            optim.zero_grad()
            pred = model(input, tgt,tgt_mask=tgt_mask)

            # Loss on normalized values
            train_loss = criterion(pred, target)
            
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step_and_update_lr()

            # Calculate ADE/FDE
            obs_last_pos = obs_tensor[:, -1:, 0:2]  # Last observed position
            mad, fad = calculate_metrics(pred.detach(), target, obs_last_pos, 
                                       True, mean, std, args.device)
            
            epoch_loss += train_loss.item()
            epoch_ade += mad
            epoch_fde += fad
            
            # Update progress bar
            load_train.set_postfix({
                'loss': f"{train_loss.item():.4f}",
                'ADE': f"{mad:.4f}",
                'FDE': f"{fad:.4f}"
            })

        # Average training metrics
        avg_train_loss = epoch_loss / len(train_dl)
        avg_train_ade = epoch_ade / len(train_dl)
        avg_train_fde = epoch_fde / len(train_dl)
        
        # Test evaluation
        model.eval()
        test_loss = 0
        test_ade = 0
        test_fde = 0
        
        with torch.no_grad():
            for batch in test_dl:
                obs_tensor, target_tensor = batch
                obs_tensor = obs_tensor.to(args.device)
                target_tensor = target_tensor.to(args.device)

                input = (obs_tensor[:,1:,2:4] - mean[2:])/std[2:]
                updated_enq_length = input.shape[1]
                target = (target_tensor[:,:,2:4] - mean[2:])/std[2:]

                tgt = torch.zeros_like(target).to(args.device)
                tgt[:, 1:, :] = target[:, :-1, :]

                tgt_mask = generate_square_mask(dim_trg=dec_seq_len, 
                                              dim_src=updated_enq_length, 
                                              mask_type="tgt").to(args.device)

                pred = model(input, tgt, tgt_mask=tgt_mask)
                
                # Calculate metrics
                loss = criterion(pred, target)
                obs_last_pos = obs_tensor[:, -1:, 0:2]
                mad, fad = calculate_metrics(pred, target, obs_last_pos, 
                                          True, mean, std, args.device)
                
                test_loss += loss.item()
                test_ade += mad
                test_fde += fad

        # Average test metrics
        avg_test_loss = test_loss / len(test_dl)
        avg_test_ade = test_ade / len(test_dl)
        avg_test_fde = test_fde / len(test_dl)

        # Save metrics
        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)
        train_ades.append(avg_train_ade)
        test_ades.append(avg_test_ade)
        train_fdes.append(avg_train_fde)
        test_fdes.append(avg_test_fde)

        # Print epoch results
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Train - Loss: {avg_train_loss:.4f}, ADE: {avg_train_ade:.4f}, FDE: {avg_train_fde:.4f}")
        print(f"Test  - Loss: {avg_test_loss:.4f}, ADE: {avg_test_ade:.4f}, FDE: {avg_test_fde:.4f}")

    # Plot training history
    plot_metrics(train_losses, test_losses, train_ades, test_ades, train_fdes, test_fdes)

    return model, {
        'train_losses': train_losses, 'test_losses': test_losses,
        'train_ades': train_ades, 'test_ades': test_ades,
        'train_fdes': train_fdes, 'test_fdes': test_fdes
    }

def plot_metrics(train_losses, test_losses, train_ades, test_ades, train_fdes, test_fdes):
    """Plot training metrics"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(test_losses, label='Test Loss')
    ax1.set_title('Loss')
    ax1.legend()
    
    # ADE plot
    ax2.plot(train_ades, label='Train ADE')
    ax2.plot(test_ades, label='Test ADE')
    ax2.set_title('ADE')
    ax2.legend()
    
    # FDE plot
    ax3.plot(train_fdes, label='Train FDE')
    ax3.plot(test_fdes, label='Test FDE')
    ax3.set_title('FDE')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()
# def train_attn(args,train_dl,test_dl,model=None,optim=None, epochs=10,enc_seq = 8,real_coords =False,dec_seq=12,mean=torch.tensor([0.0, 0.0,0.0,0.0]), std=torch.tensor([1.0, 1.0,1.0,1.0])):
#     """
#     Training loop for sequence prediction model.
    
#     Args:
#         model: PyTorch model
#         train_dl: Training data loader
#         val_dl: Validation data loader 
#         optimizer: PyTorch optimizer
#         criterion: Loss function
#         args: Training arguments
#         epochs: Number of epochs
#     """
#     print('Training Settings:')
#     print(f"Train batch size: {args.batch_size}")

#     mean = mean.to(args.device)
#     std = std.to(args.device)

#     criterion = nn.MSELoss()

#     model.train() 
#     for epoch in range (epochs):
#         # Initialize metrics
#         gts_train, preds_train = [], []
#         epoch_loss, epoch_val_loss = 0, 0
#         val_epoch_mad, val_epoch_fad = [], []


#         load_train = tqdm(train_dl, desc=f"Epoch: {epoch+1}/{epochs}")

#         # model.train()
#         for id_b,batch in enumerate(load_train):

#             obs_tensor, target_tensor = batch
#             batch_size, enc_seq_len, feat_dim = obs_tensor.shape
#             dec_seq_len = target_tensor.shape[1]
            
#             # Move to device
#             obs_tensor = obs_tensor.to(args.device)
#             target_tensor = target_tensor.to(args.device)


#             # Normalize input (seq_len-1,2) output (seq_len,2)
#             input = (obs_tensor[:,1:,2:4] - mean[2:])/std[2:]
#             target = (target_tensor[:,1:,2:4] - mean[2:])/std[2:]

#             # Create tgt with shifted target sequence 
#             tgt = torch.zeros_like(target).to(args.device)  
#             tgt[:, 1:, :] = target[:, :-1, :] 

#             tgt_mask = generate_square_mask(dim_trg = dec_seq_len ,dim_src = enc_seq_len, mask_type="tgt").to(args.device)

#             optim.zero_grad()
            

#             pred = model(input,tgt,tgt_mask = tgt_mask)

#             # Denormalize predictions and targets
#             pred_denorm = pred * std[2:] + mean[2:]
#             target_denorm = target * std[2:] + mean[2:]

#             # Compute loss
#             train_loss = criterion(pred, target)
            
#             # Backward pass
#             train_loss.backward()
            
#             # Gradient clipping
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
#             # Update the learning rate schedule and weights
#             optim.step_and_update_lr()

#             epoch_loss += train_loss.item()

            



#         #     if real_coords:
#         #          y = (batch['trg'][:, :, 0:2]).to(device)
#         #     if (add_features):
                
#         #         input_c = torch.sqrt(torch.square(batch['src'][:,1:,2].to(device)) + torch.square(batch['src'][:,1:,3].to(device))).unsqueeze(-1)
#         #         #input_d = (batch['src'][:,1:,3].to(device) / batch['src'][:,1:,2].to(device)).unsqueeze(-1)
#         #         input = torch.cat((inp,input_c),-1)
#         #         # input = torch.cat((input_temp,input_d),-1)
#         #         # print("input.shape: ", input.shape)
#         #     else:
#         #         input = inp
#         #         target = target


#         #     target_c=torch.zeros((target.shape[0],target.shape[1],1)).to(device)
#         #     target=torch.cat((target,target_c),-1)
#         #     tgt = torch.Tensor([0, 0]).unsqueeze(0).unsqueeze(1).repeat(target.shape[0],12,1).to(device)

#         #     tgt_mask = generate_square_mask(dim_trg = decoder_seq_len ,dim_src = encoder_seq_len, mask_type="tgt").to(device)
#         #     #tgt_mask = subsequent_mask(decoder_seq_len).to(device)

#         #     optim.zero_grad()
#         #     pi, sigma_x,sigma_y, mu_x , mu_y,decoder_out = model(input,tgt,tgt_mask = tgt_mask)
#         #     mus = torch.cat((mu_x.unsqueeze(-1),mu_y.unsqueeze(-1)),-1)
#         #     sigmas = torch.cat((sigma_x.unsqueeze(-1),sigma_y.unsqueeze(-1)),-1)
#         #     # print("MUS: ",mus.shape, "\nPI: ",pi.shape)
#         #     # print("MUS: ",mus[0][0], "\nPI: ",pi[0][0])

#         #     batch_pred = sample_mean(pi,sigmas,mus)

#         #     if loss_mode=='pair_wise':
#         #         # remeber decoder_out is not being optimized but sigmas and mus
#         #         loss = F.pairwise_distance(decoder_out[:, :,0:2].contiguous().view(-1, 2),((batch['trg'][:, :, 2:4].to(device)-mean.to(device))/std.to(device)).contiguous().view(-1, 2).to(device)).mean() + torch.mean(torch.abs(decoder_out[:,:,2]))
               
#         #     elif(loss_mode=='msq'):
#         #         # remeber decoder_out is not being optimized but sigmas and mus
#         #         pred = decoder_out[:, :,0:2].contiguous().view(-1, 2)
#         #         y = ((batch['trg'][:, :, 2:4].to(device)-mean.to(device))/std.to(device)).contiguous().view(-1, 2).to(device) 
#         #         loss = nn.MSELoss(pred,y)
#         #     elif(loss_mode=='mdn'):
#         #         #loss_mdn 
#         #         train_loss = mdn_loss_fn(pi, sigma_x,sigma_y, mu_x , mu_y,y,mixtures,device)
                
#         #     elif(loss_mode=='combined'):

#         #         loss_mdn = mdn_loss_fn(pi, sigma_x,sigma_y, mu_x , mu_y,y,mixtures,device)
#         #         msq = Mean_squared_distance(y.contiguous().view(-1, 2).to(device),batch_pred.to(device).contiguous().view(-1, 2))
#         #         train_loss =  model.mdn_weight * loss_mdn + (1- model.mdn_weight)*msq
#         #         #print("loss_mdn: ",loss_mdn,"   loss_mdn: ",msq ," after: ",(model.mdn_weight * loss_mdn),(loss_mdn + (1- model.mdn_weight)*msq))
                
    
            
#         #     #print("shape: ",batch_pred.contiguous().view(-1, 2).shape)
#         #     train_loss.backward()
#         #     # Update the learning rate schedule
#         #     optim.step_and_update_lr()

#         #     epoch_loss += train_loss.item()
        
#         # avg_loss_epoch = epoch_loss / num_batches
#         # loss_list.append(avg_loss_epoch)
        
#         # lr = optim._optimizer.param_groups[0]['lr']

        


#     #     ## EVALUATE in validation

#     #     with torch.no_grad():
#     #         model.eval()
#     #         gts_ev,preds_ev,src_ev = [], [] ,[]

#     #         for id_e, val_batch in enumerate(val_dl):
#     #             #load_val.set_description(f"Epoch: {epoch+1} / {epochs}")
#     #             src_ev.append(val_batch['src'])
#     #             gts_ev.append(val_batch['trg'][:, :, 0:2])
#     #             batch_gt_val = val_batch['trg'][:, :, 0:2].to(device)

#     #             if(normalized):
#     #                 inp_val=(val_batch['src'][:,1:,2:4].to(device)-mean.to(device))/std.to(device)
#     #                 target_val=(val_batch['trg'][:,:-1,2:4].to(device)-mean.to(device))/std.to(device)
#     #                 y_val = (val_batch['trg'][:, :, 2:4].to(device)-mean.to(device))/std.to(device)

#     #                 # input_valc = torch.sqrt(torch.square(val_batch['src'][:,1:,2].to(device)) + torch.square(val_batch['src'][:,1:,3].to(device))).unsqueeze(-1)

#     #                 # input_val = torch.cat((inp_val,input_valc),-1)
#     #             else:
#     #                 inp_val = val_batch['src'][:,1:,2:4].to(device)
#     #                 target_val = val_batch['trg'][:,:-1,2:4].to(device)
#     #                 y_val = val_batch['trg'][:, :, 2:4].to(device)
#     #                 batch_gt_val = val_batch['trg'][:, :, 0:2].to(device)
                
#     #             if real_coords:
#     #                 y_val = (val_batch['trg'][:, :, 0:2]).to(device)

#     #             if (add_features):                

#     #                 input_valc = torch.sqrt(torch.square(val_batch['src'][:,1:,2].to(device)) + torch.square(val_batch['src'][:,1:,3].to(device))).unsqueeze(-1)
#     #                 input_val = torch.cat((inp_val,input_valc),-1)
#     #             else:
#     #                 input_val = inp_val
#     #                 target_val = target_val


#     #             tgt_val = torch.Tensor([0, 0]).unsqueeze(0).unsqueeze(1).repeat(target_val.shape[0],12,1).to(device)

               
                
#     #             tgt_val_mask = generate_square_mask(dim_trg = decoder_seq_len ,dim_src = encoder_seq_len, mask_type="tgt").to(device)

#     #             pi_val, sigma_x_val,sigma_y_val, mu_x_val , mu_y_val,decoder_out = model(input_val,tgt_val,tgt_mask = tgt_val_mask)

#     #             mus_val = torch.cat((mu_x_val.unsqueeze(-1),mu_y_val.unsqueeze(-1)),-1)
#     #             sigmas_val = torch.cat((sigma_x_val.unsqueeze(-1),sigma_y_val.unsqueeze(-1)),-1)

#     #             batch_pred_val = sample_mean(pi_val,sigmas_val,mus_val)

#     #             #params = [pi, sigma_x,sigma_y, mu_x , mu_y,decoder_out]

#     #             if(loss_mode=='mdn'):
#     #                 #loss_mdn 
#     #                 loss_val = mdn_loss_fn(pi_val, sigma_x_val,sigma_y_val, mu_x_val , mu_y_val,y_val,mixtures,device)
                    
#     #             elif(loss_mode=='combined'):
#     #                 loss_val_mdn = mdn_loss_fn(pi_val, sigma_x_val,sigma_y_val, mu_x_val , mu_y_val,y_val,mixtures,device)
#     #                 msq_val = Mean_squared_distance(y_val.contiguous().view(-1, 2).to(device),batch_pred_val.contiguous().view(-1, 2).to(device))

#     #                 loss_val =  model.mdn_weight * loss_val_mdn + (1- model.mdn_weight)*msq_val
#     #                 #print("loss_mdn: ",loss_mdn,"   loss_mdn: ",msq ," after: ",(model.mdn_weight * loss_mdn),(loss_mdn + (1- model.mdn_weight)*msq))

#     #             epoch_val_loss += loss_val.item()

#     #             batch_gt_val = batch_gt_val.detach().cpu().numpy()
            
#     #             if (post_process and normalized):
#     #             #print("TRUE")
#     #                 batch_pred_val = (batch_pred_val[:, :,:] * std.to(device) + mean.to(device)).cpu().numpy().cumsum(1) + val_batch[
#     #                                                                                                             'src'][
#     #                                                                                                         :, -1:,
#     #                                                                                                         0:2].cpu().numpy()
#     #                 batch_gt_val = val_batch['trg'][:, :, 0:2].to(device).cpu().numpy()
#     #             else:
#     #                 batch_pred_val = batch_pred_val.detach().cpu().numpy()
#     #                 batch_pred_val = batch_pred_val[:, :,:].cumsum(1) + val_batch['src'][:, -1:,0:2].cpu().numpy()
#     #             # calcualte error:
 
#     #             mad, fad, errs = distance_metrics(batch_gt_val, batch_pred_val)
#     #             val_epoch_mad.append(mad)
#     #             val_epoch_fad.append(fad)

        
#     #     avg_loss_epoch_val = epoch_val_loss / num_batches_val
#     #     sum(val_epoch_mad)
#     #     val_batch_mad = (sum(val_epoch_mad)/num_batches_val)
#     #     val_batch_fad = (sum(val_epoch_fad)/num_batches_val)
#     #     val_mad.append(val_batch_mad)
#     #     val_fad.append(val_batch_fad)
#     #     loss_eval.append(avg_loss_epoch_val)

#     #     # save and stop model
#     #     early_stop(model,avg_loss_epoch, avg_loss_epoch_val,epoch+1,val_batch_mad,val_batch_fad)
#     #     print(f"Train loss:{avg_loss_epoch:.4f} mdn weight: {model.mdn_weight:.3f}") #mdn weighte: {model.mdn_weight}
#     #     print(f"Eval loss: {avg_loss_epoch_val:.4f}")
#     #     print(f"Eval val_batch_mad: {val_batch_mad:.4f}")
#     #     print(f"Eval val_batch_fad: {val_batch_fad:.4f}")
#     #     print(f"Learning Rate: {lr:.5f}")
        
#     #     #mad_test , fad_test,_,_,mdn_results,avg_mad,avg_fad= test_mdn(test_dl, model,device,add_features = add_features,mixtures=mixtures,enc_seq = 8,dec_seq=12, mode='feed',loss_mode ='mdn',mean=mean,std=std)
#     #     # test_mad.append(avg_mad)
#     #     # test_fad.append(avg_fad)
#     #     # Test the model
#     #     if Args.show_test:
#     #         print('----- Test -----')
#     #         batch_preds,batch_gts,avg_mad,avg_fad,candidate_trajs,candidate_weights,best_candiates,src_trajs = test_mdn(test_dl, model,device,add_features = add_features,mixtures=mixtures,enc_seq = 8,dec_seq=12, mode='feed',loss_mode ='mdn',mean=mean,std=std)
#     #         print('----- END -----')

#     #     # if (early_stop.stop):
#     #     #     print("Early stopping activated!")
#     #     #     break

        

#     # print(f"Epoch {epoch+1} Train loss: {avg_loss_epoch}")
#     # print(f"Epoch {epoch+1} Eval loss: {avg_loss_epoch_val}")
#     # return loss_list, loss_eval,val_mad,val_fad#all_mad,all_fad,loss_list



