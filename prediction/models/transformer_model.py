import torch
import torch.nn as nn
import math
import torch.optim as optim
import numpy as np
import math
from torch.autograd import Variable # storing data while learning
from config import CFG

from utils import load_data,Dataset,create_tgt,generate_square_mask,PositionalEncoding


device = CFG.device
batch_size = CFG.batch_size


class Attention_EMT(nn.Module):
    def __init__(self,
        in_features = 2,
        out_features = 2,
        num_heads = 2,
        num_encoder_layers = 3,
        num_decoder_layers = 3,
        embedding_size = 128,
        dropout = 0.1,
        max_length = 12,
        batch_first = True,
        actn = "gelu"
        ):
            super(Attention_EMT, self).__init__()
            # self.device = device
            self.num_heads = num_heads
            self.num_encoder_layers = num_encoder_layers
            self.num_decoder_layers = num_decoder_layers
            self.max_len = max_length
            self.input_features = in_features
            self.output_features = out_features
            self.out_length = max_length
            self.d_model= embedding_size # selected
            self.dropout_encoder = dropout
            self.dropout_decoder = dropout
            self.dropout_pos_enc = 0.0
            self.ndim = 2
        
            
            #Following the original transformer paper's design where dim_feedforward is typically 4x larger than d_model
            self.dim_feedforward_encoder = 4*self.d_model
            self.dim_feedforward_decoder = 4*self.d_model


            # Positional Encoding
            self.positional_encoding_layer = PositionalEncoding(
                d_model=self.d_model,
                dropout=self.dropout_pos_enc,
                max_len = self.max_len,
                batch_first = batch_first
                )

            # Creating the  linear layers needed for the model
            self.encoder_input_layer = Linear_Embeddings(self.input_features, self.d_model) 
            self.decoder_input_layer = Linear_Embeddings(self.output_features, self.d_model)   


            self.encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.num_heads, 
                dim_feedforward=self.dim_feedforward_encoder,
                dropout=self.dropout_encoder,
                batch_first=batch_first,
                activation=actn
                )


            self.encoder = nn.TransformerEncoder(
                encoder_layer =  self.encoder_layer,
                num_layers = self.num_encoder_layers
                )
            
            # Create the decoder layer
            self.decoder_layer = nn.TransformerDecoderLayer(
                d_model=self.d_model,
                nhead=self.num_heads, 
                dim_feedforward = self.dim_feedforward_decoder,
                dropout = self.dropout_decoder,
                batch_first=batch_first,   
                activation=actn
            )

            self.decoder = nn.TransformerDecoder(
                decoder_layer = self.decoder_layer,
                num_layers=self.num_decoder_layers
                )
            
            # Output layer (to map decoder output to the target shape)
            self.output_layer = nn.Linear(self.d_model, self.output_features)  

    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask: torch.Tensor=None, 
                tgt_mask: torch.Tensor=None) -> torch.Tensor:
        
        # Move inputs to device at the start
        src = src.to(device)
        tgt = tgt.to(device)
        if src_mask is not None:
            src_mask = src_mask.to(device)
        if tgt_mask is not None:
            tgt_mask = tgt_mask.to(device)

        # Embedding 
        encoder_embed = self.encoder_input_layer(src)
        encoder_embed = self.positional_encoding_layer(encoder_embed) 

        # src shape: [batch_size, enc_seq_len, dim_val]
        encoder_out = self.encoder(src=encoder_embed)

        # Pass decoder input through decoder input layer
        decoder_embed = self.decoder_input_layer(tgt)
        decoder_output = self.positional_encoding_layer(decoder_embed) 

        # Pass throguh decoder - output shape: [batch_size, target seq len, dim_val]
        decoder_output = self.decoder(
            tgt=decoder_output,
            memory=encoder_out,
            tgt_mask=tgt_mask,
            memory_mask=src_mask
            )
        
        # Output projection shape: [batch_size, target seq len, dim_val]
        output = self.output_layer(decoder_output)

        return  output

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        # lut => lookup table
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
class Linear_Embeddings(nn.Module):
    def __init__(self, input_features,d_model):
        super(Linear_Embeddings, self).__init__()
        # lut => lookup table
        self.lut = nn.Linear(input_features, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
class AttentionMDN_Old(nn.Module):
    def __init__(self,
        device,
        num_features,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        dropout_p,
        n_gaussians = 5,
        n_hidden = 10,
        max_length = 8,
        batch_first = True
        ):
            super(AttentionMDN_Old, self).__init__()
            self.device = device
            self.num_heads = num_heads
            self.num_encoder_layers = num_encoder_layers
            self.num_decoder_layers = num_decoder_layers
            self.max_len = max_length
            self.input_features = num_features
            self.output_features = num_features
            self.dim_feedforward_encoder = 2048
            self.dim_feedforward_decoder = 2048
            self.out_length = max_length
            self.d_model= 512 # selected
            self.dropout_encoder = 0.2
            self.dropout_decoder = 0.2
            self.dropout_pos_enc = 0.1
            # self.dropout = dropout_p
            self.gaussians =  n_gaussians
            self.hidden = n_hidden
            self.flatten = nn.Flatten(start_dim=1)
            self.ndim = 2
            self.relu = nn.ReLU()
            self.mdn_input_shape = self.output_features * self.max_len # input features*seq_length

            
            # Positional Encoding
            self.positional_encoding_layer = PositionalEncoding(
                d_model=self.d_model,
                dropout=self.dropout_pos_enc,
                max_len = self.max_len,
                batch_first = batch_first
                )
            #self.pos_encoder = self.positional_encoding(self.max_len, self.d_model).to(device)

            # Creating the  linear layers needed for the model
            self.encoder_input_layer = nn.Linear(
                in_features= self.input_features , 
                out_features= self.d_model 
                )

            self.decoder_input_layer = nn.Linear(
                in_features = self.output_features,
                out_features = self.d_model 
                )  

            # Stack the encoder layer n times in nn.TransformerDecoder
            # The encoder layer used in the paper is identical to the one used by
            # Vaswani et al (2017) on which the PyTorch module is based.
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.num_heads, 
                dim_feedforward=self.dim_feedforward_encoder,
                dropout=self.dropout_encoder,
                batch_first=batch_first
                )


            # Stack the encoder layers in nn.TransformerDecoder
            # It seems the option of passing a normalization instance is redundant
            # in my case, because nn.TransformerEncoderLayer per default normalizes
            # after each sub-layer
            # (https://github.com/pytorch/pytorch/issues/24930).
            self.encoder = nn.TransformerEncoder(
                encoder_layer = encoder_layer,
                num_layers = self.num_encoder_layers, 
                norm=None
                )
            # Create the decoder layer
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=self.d_model,
                nhead=self.num_heads, 
                dim_feedforward = self.dim_feedforward_decoder,
                dropout = self.dropout_decoder,
                batch_first=batch_first
            )

            # Stack the decoder layers in nn.TransformerDecoder
            # It seems the option of passing a normalization instance is redundant
            # in my case, because nn.TransformerDecoderLayer per default normalizes
            # after each sub-layer
            # (https://github.com/pytorch/pytorch/issues/24930).
            self.decoder = nn.TransformerDecoder(
                decoder_layer = decoder_layer,
                num_layers=self.num_decoder_layers, 
                norm=None
                )

            self.out = nn.Linear(self.d_model, self.output_features)

            # Mixed Density Network 

            # if the input of linear is associated with out_features, then the number of nodes will be 2
            # This is not useful coz it learns relation of x and y only
            # instead relation of sequence should also be learned (batch_size,-1) the -1 indicates feature_size*seq_length

            self.z_h = nn.Sequential(
            nn.Linear(self.mdn_input_shape,self.hidden),
            #nn.Sigmoid()
            nn.LeakyReLU(0.1)
            )

            self.z_pi_old = nn.Linear(self.hidden, self.gaussians).to(device)
            self.z_sigma_old = nn.Linear(self.hidden, self.gaussians*self.ndim).to(device)
            self.z_mu_old = nn.Linear(self.hidden, self.gaussians*self.ndim).to(device)  

            self.z_pi = nn.Linear(self.hidden,self.gaussians).to(device)
            self.z_sigma = nn.Linear(self.hidden, self.gaussians*self.ndim).to(device)
            self.z_mu = nn.Linear(self.hidden, self.gaussians*self.ndim).to(device)

            #self.z_sigma = nn.Linear(self.hidden, self.out_length*self.gaussians*self.ndim).to(device) 
    

    # def positional_encoding(self, max_len, d_model):
    #     pos = torch.arange(0, max_len).unsqueeze(1)
    #     i = torch.arange(0, d_model, 2).float()
    #     div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    #     pe = pos * div_term.unsqueeze(0)
    #     pe[:, 0::2] = torch.sin(pe[:, 0::2])
    #     pe[:, 1::2] = torch.cos(pe[:, 1::2])
    #     return pe.unsqueeze(0)
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask: torch.Tensor=None, 
                tgt_mask: torch.Tensor=None) -> torch.Tensor:
        # Embedding 
        src = self.encoder_input_layer(src).to(device)
        # print("Starting ...")

        src = self.positional_encoding_layer(src).to(device) 


        # print("Source and Target positional encoding Done!")
        # print("Transformer Started ...",src.shape)

        src = self.encoder( # src shape: [batch_size, enc_seq_len, dim_val]
        src=src
        ).to(device)

        # Pass decoder input through decoder input layer
        decoder_output = self.decoder_input_layer(tgt).to(device)

        # Pass throguh decoder - output shape: [batch_size, target seq len, dim_val]
        decoder_output = self.decoder(
            tgt=decoder_output,
            memory=src,
            tgt_mask=tgt_mask,
            memory_mask=src_mask
            ).to(device)

        decoder_output = self.out(decoder_output).to(device)



        # transformer_out = self.transformer(src, tgt).to(device)
        #print("Transformer  Finished!")

        #print("decoder_output shape: ",decoder_output.shape)

        #x = 
        # x = self.flatten(decoder_output).to(device)
        #print("x shape: ",x.shape)
        z_h = self.z_h(decoder_output).to(device)
        #print("z_h shape: ",z_h.shape)

        # Calculate PI
        pi = self.z_pi(z_h).to(device)
        # print("pi before shape: ",pi.shape)
        # Reshape Pi before softmax 
        # pi = pi.view(-1,self.out_length,self.gaussians)
        # print("pi honest shape: ",pi.shape)
        # Softmax for probablistic output
        pi = nn.functional.softmax(pi, -1)
        # print("pi final shape: ",pi.shape)
        
        # Calculate Sigma and Mu
        sigma = torch.exp(self.z_sigma(z_h)).to(device)
        mu = self.z_mu(z_h).to(device)
    

        sigma_x = sigma[:,:,:self.gaussians]
        sigma_y = sigma[:,:,self.gaussians:]
        mu_x = mu[:,:,:self.gaussians]
        mu_y = mu[:,:,self.gaussians:]

        return  pi, sigma_x,sigma_y, mu_x ,mu_y,decoder_output

class Transformer_MDN(nn.Module):
    def __init__(self,
        device,
        in_features,
        out_features,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        embedding_size,
        n_gaussians = 5,
        n_hidden = 10,
        max_length = 12,
        batch_first = True,
        actn = "gelu"
        ):
            super(Transformer_MDN, self).__init__()
            self.device = device
            self.num_heads = num_heads
            self.num_encoder_layers = num_encoder_layers
            self.num_decoder_layers = num_decoder_layers
            self.max_len = max_length
            self.input_features = in_features
            self.output_features = out_features
            self.dim_feedforward_encoder = 2048
            self.dim_feedforward_decoder = 2048
            self.out_length = max_length
            self.d_model= embedding_size # selected
            self.dropout_encoder = 0.2
            self.dropout_decoder = 0.2
            self.dropout_pos_enc = 0.1
            # self.dropout = dropout_p
            self.gaussians =  n_gaussians
            self.hidden = n_hidden
            self.ndim = 2
        


            # Positional Encoding
            self.positional_encoding_layer = PositionalEncoding(
                d_model=self.d_model,
                dropout=self.dropout_pos_enc,
                max_len = self.max_len,
                batch_first = batch_first
                )

            #self.pos_encoder = self.positional_encoding(self.max_len, self.d_model).to(device)



            # Creating the  linear layers needed for the model
            # self.encoder_input_layer = nn.Linear(
            #     in_features= self.input_features , 
            #     out_features= self.d_model 
            #     )
            
            # self.encoder_input_layer = nn.Linear(
            #     in_features= self.input_features , 
            #     out_features= self.d_model 
            #     )

            self.encoder_input_layer = Linear_Embeddings(self.input_features, self.d_model) 
            self.decoder_input_layer = Linear_Embeddings(self.output_features, self.d_model)   

            # # Layer Normalization
            # self.norm_layer = nn.LayerNorm(self.d_model)

            # Stack the encoder layer n times in nn.TransformerDecoder
            # The encoder layer used in the paper is identical to the one used by
            # Vaswani et al (2017) on which the PyTorch module is based.
            self.encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.num_heads, 
                dim_feedforward=self.dim_feedforward_encoder,
                dropout=self.dropout_encoder,
                batch_first=batch_first,
                activation=actn
                )


            self.encoder = nn.TransformerEncoder(
                encoder_layer =  self.encoder_layer,
                num_layers = self.num_encoder_layers
                )
            # Create the decoder layer
            self.decoder_layer = nn.TransformerDecoderLayer(
                d_model=self.d_model,
                nhead=self.num_heads, 
                dim_feedforward = self.dim_feedforward_decoder,
                dropout = self.dropout_decoder,
                batch_first=batch_first,   
                activation=actn
            )

            self.decoder = nn.TransformerDecoder(
                decoder_layer = self.decoder_layer,
                num_layers=self.num_decoder_layers
                )

            self.out = nn.Linear(self.d_model, self.output_features)

    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask: torch.Tensor=None, 
                tgt_mask: torch.Tensor=None) -> torch.Tensor:
        # Embedding 
        encoder_embed = self.encoder_input_layer(src).to(device)
        encoder_embed = self.positional_encoding_layer(encoder_embed).to(device) 

        # src shape: [batch_size, enc_seq_len, dim_val]
        encoder_out = self.encoder(src=encoder_embed).to(device)

        # Pass decoder input through decoder input layer
        decoder_embed = self.decoder_input_layer(tgt).to(device)
        decoder_output = self.positional_encoding_layer(decoder_embed).to(device) 

        # Pass throguh decoder - output shape: [batch_size, target seq len, dim_val]
        decoder_output = self.decoder(
            tgt=decoder_output,
            memory=encoder_out,
            tgt_mask=tgt_mask,
            memory_mask=src_mask
            ).to(device)

        decoder_output = self.out(decoder_output).to(device)

        result = torch.Tensor(decoder_output)

        return  result

class Attention_GMM(nn.Module):
    def __init__(self,
        in_features,
        out_features,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        embedding_size,
        n_gaussians = 5,
        n_hidden = 10,
        dropout=0.2,
        max_length = 12,
        batch_first = True,
        actn = "gelu"
        ):
            super(Attention_GMM, self).__init__()
            # self.device = device
            self.num_heads = num_heads
            self.num_encoder_layers = num_encoder_layers
            self.num_decoder_layers = num_decoder_layers
            self.max_len = max_length
            self.input_features = in_features
            self.output_features = out_features
            self.dim_feedforward_encoder = 2048
            self.dim_feedforward_decoder = 2048
            self.out_length = max_length
            self.d_model= embedding_size # selected
            self.dropout_encoder = dropout
            self.dropout_decoder = dropout
            self.dropout_pos_enc = dropout
            # self.dropout = dropout_p
            self.gaussians =  n_gaussians
            self.hidden = n_hidden
            self.ndim = 2
            self.mdn_weight =torch.tensor(0.5)
            # self.mdn_weight = nn.Parameter(torch.tensor(0.5),requires_grad=True)
            #self.mdn_weight = nn.Parameter(torch.tensor([0.5]),requires_grad=True).to(device)
        


            # Positional Encoding
            self.positional_encoding_layer = PositionalEncoding(
                d_model=self.d_model,
                dropout=self.dropout_pos_enc,
                max_len = self.max_len,
                batch_first = batch_first
                )

            # Creating the  linear layers needed for the model

            self.encoder_input_layer = Linear_Embeddings(self.input_features, self.d_model) 
            self.decoder_input_layer = Linear_Embeddings(self.output_features, self.d_model)   


            # Stack the encoder layer n times in nn.TransformerDecoder
            # The encoder layer used in the paper is identical to the one used by
            # Vaswani et al (2017) on which the PyTorch module is based.
            self.encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.num_heads, 
                dim_feedforward=self.dim_feedforward_encoder,
                dropout=self.dropout_encoder,
                batch_first=batch_first,
                activation=actn
                )


            self.encoder = nn.TransformerEncoder(
                encoder_layer =  self.encoder_layer,
                num_layers = self.num_encoder_layers
                )
            # Create the decoder layer
            self.decoder_layer = nn.TransformerDecoderLayer(
                d_model=self.d_model,
                nhead=self.num_heads, 
                dim_feedforward = self.dim_feedforward_decoder,
                dropout = self.dropout_decoder,
                batch_first=batch_first,   
                activation=actn
            )

            self.decoder = nn.TransformerDecoder(
                decoder_layer = self.decoder_layer,
                num_layers=self.num_decoder_layers
                )


            self.embedding_sigma = nn.Sequential(
            nn.Linear(self.d_model,self.hidden),
            nn.ELU(),#nn.GELU(),#nn.LeakyReLU(), #nn.GELU(),#nn.ReLU(),
            nn.Linear(self.hidden,self.hidden//2),
            nn.ELU(),
            nn.Linear(self.hidden//2,self.hidden//4),
            nn.ELU()#nn.GELU(),#nn.LeakyReLU(),#nn.GELU(),#nn.ReLU(),
            )
            self.embedding_mue = nn.Sequential(
            nn.Linear(self.d_model,self.hidden),
            nn.ELU(),#nn.GELU(),#nn.LeakyReLU(), #nn.GELU(),#nn.ReLU(),
            nn.Linear(self.hidden,self.hidden//2),
            nn.ELU(),
            nn.Linear(self.hidden//2,self.hidden//4),
            nn.ELU()#nn.GELU(),#nn.LeakyReLU(),#nn.GELU(),#nn.ReLU(),
            )

            self.pis = nn.Sequential(
            nn.Linear(self.d_model,self.hidden),
            nn.ELU(),
            nn.Linear(self.hidden,self.hidden//2),
            nn.ELU(),#nn.GELU(),#nn.LeakyReLU(), #nn.GELU(),#nn.ReLU(),
            nn.Linear(self.hidden//2,self.hidden//4),
            nn.ELU(),
            nn.Linear(self.hidden//4,self.gaussians)
            #nn.Softmax()
            )

            
            self.hidden_hid = self.hidden//4
            # self.pis = nn.Linear(self.hidden_hid,self.gaussians).to(device)
            self.sigma_x = nn.Linear(self.hidden_hid, self.gaussians)
            self.sigma_y = nn.Linear(self.hidden_hid, self.gaussians)
            self.mu_x = nn.Linear(self.hidden_hid, self.gaussians)
            self.mu_y = nn.Linear(self.hidden_hid, self.gaussians)

    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask: torch.Tensor=None, 
                tgt_mask: torch.Tensor=None) -> torch.Tensor:
        # Embedding 
        encoder_embed = self.encoder_input_layer(src).to(device)
        encoder_embed = self.positional_encoding_layer(encoder_embed).to(device) 

        # src shape: [batch_size, enc_seq_len, dim_val]
        encoder_out = self.encoder(src=encoder_embed).to(device)

        # Pass decoder input through decoder input layer
        decoder_embed = self.decoder_input_layer(tgt).to(device)
        decoder_output = self.positional_encoding_layer(decoder_embed).to(device) 

        # Pass throguh decoder - output shape: [batch_size, target seq len, dim_val]
        decoder_output = self.decoder(
            tgt=decoder_output,
            memory=encoder_out,
            tgt_mask=tgt_mask,
            memory_mask=src_mask
            ).to(device)

        #mdn_embeded = self.embedding_mdn(decoder_output).to(device)
        sigmax_embeded = self.embedding_sigma(decoder_output).to(device)
        sigmay_embeded = self.embedding_sigma(decoder_output).to(device)
        muex_embeded = self.embedding_mue(decoder_output).to(device)
        muey_embeded = self.embedding_mue(decoder_output).to(device)

        # Calculate PI
        pi = self.pis(decoder_output).to(device)
        pi = nn.functional.softmax(pi, -1)

        # Calculate Sigmas
        sigma_x = torch.Tensor(torch.exp(self.sigma_x(sigmax_embeded))).to(device)
        sigma_y = torch.Tensor(torch.exp(self.sigma_x(sigmay_embeded))).to(device)

        mu_x = torch.Tensor(self.mu_x(muex_embeded)).to(device)
        mu_y = torch.Tensor(self.mu_y(muey_embeded)).to(device)
       
        #result = torch.Tensor(decoder_output)

        return  pi, sigma_x,sigma_y, mu_x ,mu_y,decoder_output
    
class Attention_GMM_Encoder(nn.Module):
    def __init__(self,
        device,
        in_features,
        out_features,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        embedding_size,
        n_gaussians = 5,
        n_hidden = 10,
        dropout=0.2,
        max_length = 12,
        batch_first = True,
        actn = "gelu"
        ):
            super(Attention_GMM_Encoder, self).__init__()
            self.device = device
            self.num_heads = num_heads
            self.num_encoder_layers = num_encoder_layers
            self.num_decoder_layers = num_decoder_layers
            self.max_len = max_length
            self.input_features = in_features
            self.output_features = out_features
            self.dim_feedforward_encoder = 2048
            self.dim_feedforward_decoder = 2048
            self.out_length = max_length
            self.d_model= embedding_size # selected
            self.dropout_encoder = dropout
            self.dropout_decoder = dropout
            self.dropout_pos_enc = dropout
            # self.dropout = dropout_p
            self.gaussians =  n_gaussians
            self.hidden = n_hidden
            self.ndim = 2
            #self.mdn_weight = nn.Parameter(torch.tensor([.5]))
            
        


            # Positional Encoding
            self.positional_encoding_layer = PositionalEncoding(
                d_model=self.d_model,
                dropout=self.dropout_pos_enc,
                max_len = self.max_len,
                batch_first = batch_first
                )

            # Creating the  linear layers needed for the model

            self.encoder_input_layer = Linear_Embeddings(self.input_features, self.d_model) 
            self.decoder_input_layer = Linear_Embeddings(self.output_features, self.d_model)   


            # Stack the encoder layer n times in nn.TransformerDecoder
            # The encoder layer used in the paper is identical to the one used by
            # Vaswani et al (2017) on which the PyTorch module is based.
            self.encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.num_heads, 
                dim_feedforward=self.dim_feedforward_encoder,
                dropout=self.dropout_encoder,
                batch_first=batch_first,
                activation=actn
                )


            self.encoder = nn.TransformerEncoder(
                encoder_layer =  self.encoder_layer,
                num_layers = self.num_encoder_layers
                )
            # Create the decoder layer
            self.decoder_layer = nn.TransformerDecoderLayer(
                d_model=self.d_model,
                nhead=self.num_heads, 
                dim_feedforward = self.dim_feedforward_decoder,
                dropout = self.dropout_decoder,
                batch_first=batch_first,   
                activation=actn
            )

            self.decoder = nn.TransformerDecoder(
                decoder_layer = self.decoder_layer,
                num_layers=self.num_decoder_layers
                )

            #self.out = nn.Linear(self.d_model, self.output_features)
            #self.hidden_hid = self.hidden//2

            self.embedding_sigma = nn.Sequential(
            nn.Linear(self.d_model,self.hidden),
            nn.GELU(),#nn.GELU(),#nn.LeakyReLU(), #nn.GELU(),#nn.ReLU(),
            nn.Linear(self.hidden,self.self.hidden//2),
            nn.GELU(),
            nn.Linear(self.hidden//2,self.hidden//4),
            nn.GELU()#nn.GELU(),#nn.LeakyReLU(),#nn.GELU(),#nn.ReLU(),
            )
            self.embedding_mue = nn.Sequential(
            nn.Linear(self.d_model,self.hidden),
            nn.GELU(),#nn.GELU(),#nn.LeakyReLU(), #nn.GELU(),#nn.ReLU(),
            nn.Linear(self.hidden,self.self.hidden//2),
            nn.GELU(),
            nn.Linear(self.hidden//2,self.hidden//4),
            nn.GELU()#nn.GELU(),#nn.LeakyReLU(),#nn.GELU(),#nn.ReLU(),
            )

            self.pis = nn.Sequential(
            nn.Linear(self.d_model,self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden,self.hidden_hid),
            nn.GELU(),#nn.GELU(),#nn.LeakyReLU(), #nn.GELU(),#nn.ReLU(),
            nn.Linear(self.hidden_hid,self.gaussians)
            #nn.Softmax()
            ).to(device)

            # self.pis = nn.Linear(self.hidden_hid,self.gaussians).to(device)
            self.sigma_x = nn.Linear(self.hidden//4, self.gaussians).to(device)
            self.sigma_y = nn.Linear(self.hidden//4, self.gaussians).to(device)
            self.mu_x = nn.Linear(self.hidden//4, self.gaussians).to(device)
            self.mu_y = nn.Linear(self.hidden//4, self.gaussians).to(device)

    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask: torch.Tensor=None, 
                tgt_mask: torch.Tensor=None) -> torch.Tensor:
        # Embedding 
        encoder_embed = self.encoder_input_layer(src).to(device)
        encoder_embed = self.positional_encoding_layer(encoder_embed).to(device) 

        # src shape: [batch_size, enc_seq_len, dim_val]
        encoder_out = self.encoder(src=encoder_embed).to(device)

        # Pass decoder input through decoder input layer
        decoder_embed = self.decoder_input_layer(tgt).to(device)
        decoder_output = self.positional_encoding_layer(decoder_embed).to(device) 

        # Pass throguh decoder - output shape: [batch_size, target seq len, dim_val]
        decoder_output = self.decoder(
            tgt=decoder_output,
            memory=encoder_out,
            tgt_mask=tgt_mask,
            memory_mask=src_mask
            ).to(device)

        #mdn_embeded = self.embedding_mdn(decoder_output).to(device)
        # sigmax_embeded = self.embedding_sigma(decoder_output).to(device)
        # sigmay_embeded = self.embedding_sigma(decoder_output).to(device)
        # muex_embeded = self.embedding_mue(decoder_output).to(device)
        # muey_embeded = self.embedding_mue(decoder_output).to(device)

        # Calculate PI
        pi = self.pis(decoder_output).to(device)
        pi = nn.functional.softmax(pi, -1)

        # Calculate Sigmas
        sigma_embeded = self.decoder(
            tgt=decoder_output,
            memory=encoder_out,
            tgt_mask=tgt_mask,
            memory_mask=src_mask
            ).to(device)
        mu_embeded = self.decoder(
            tgt=decoder_output,
            memory=encoder_out,
            tgt_mask=tgt_mask,
            memory_mask=src_mask
            ).to(device)
        

        sigmax_embeded = self.embedding_sigma(sigma_embeded).to(device)
        sigmay_embeded = self.embedding_sigma(sigma_embeded).to(device)
        muex_embeded = self.embedding_mue(mu_embeded).to(device)
        muey_embeded = self.embedding_mue(mu_embeded).to(device)
        
        sigma_x = torch.Tensor(torch.exp(self.sigma_x(sigmax_embeded))).to(device)
        sigma_y = torch.Tensor(torch.exp(self.sigma_x(sigmay_embeded))).to(device)

        mu_x = torch.Tensor(self.mu_x(muex_embeded)).to(device)
        mu_y = torch.Tensor(self.mu_y(muey_embeded)).to(device)
       
        #result = torch.Tensor(decoder_output)

        return  pi, sigma_x,sigma_y, mu_x ,mu_y,decoder_output
    

class Attention_GMM_With_Device(nn.Module):
    def __init__(self,
        device,
        in_features,
        out_features,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        embedding_size,
        n_gaussians = 5,
        n_hidden = 10,
        dropout=0.2,
        max_length = 12,
        batch_first = True,
        actn = "gelu"
        ):
            super(Attention_GMM, self).__init__()
            self.device = device
            self.num_heads = num_heads
            self.num_encoder_layers = num_encoder_layers
            self.num_decoder_layers = num_decoder_layers
            self.max_len = max_length
            self.input_features = in_features
            self.output_features = out_features
            self.dim_feedforward_encoder = 2048
            self.dim_feedforward_decoder = 2048
            self.out_length = max_length
            self.d_model= embedding_size # selected
            self.dropout_encoder = dropout
            self.dropout_decoder = dropout
            self.dropout_pos_enc = dropout
            # self.dropout = dropout_p
            self.gaussians =  n_gaussians
            self.hidden = n_hidden
            self.ndim = 2
            self.mdn_weight =torch.tensor(0.5)
            # self.mdn_weight = nn.Parameter(torch.tensor(0.5),requires_grad=True)
            #self.mdn_weight = nn.Parameter(torch.tensor([0.5]),requires_grad=True).to(device)
        


            # Positional Encoding
            self.positional_encoding_layer = PositionalEncoding(
                d_model=self.d_model,
                dropout=self.dropout_pos_enc,
                max_len = self.max_len,
                batch_first = batch_first
                )

            # Creating the  linear layers needed for the model

            self.encoder_input_layer = Linear_Embeddings(self.input_features, self.d_model) 
            self.decoder_input_layer = Linear_Embeddings(self.output_features, self.d_model)   


            # Stack the encoder layer n times in nn.TransformerDecoder
            # The encoder layer used in the paper is identical to the one used by
            # Vaswani et al (2017) on which the PyTorch module is based.
            self.encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.num_heads, 
                dim_feedforward=self.dim_feedforward_encoder,
                dropout=self.dropout_encoder,
                batch_first=batch_first,
                activation=actn
                )


            self.encoder = nn.TransformerEncoder(
                encoder_layer =  self.encoder_layer,
                num_layers = self.num_encoder_layers
                )
            # Create the decoder layer
            self.decoder_layer = nn.TransformerDecoderLayer(
                d_model=self.d_model,
                nhead=self.num_heads, 
                dim_feedforward = self.dim_feedforward_decoder,
                dropout = self.dropout_decoder,
                batch_first=batch_first,   
                activation=actn
            )

            self.decoder = nn.TransformerDecoder(
                decoder_layer = self.decoder_layer,
                num_layers=self.num_decoder_layers
                )


            self.embedding_sigma = nn.Sequential(
            nn.Linear(self.d_model,self.hidden),
            nn.ELU(),#nn.GELU(),#nn.LeakyReLU(), #nn.GELU(),#nn.ReLU(),
            nn.Linear(self.hidden,self.hidden//2),
            nn.ELU(),
            nn.Linear(self.hidden//2,self.hidden//4),
            nn.ELU()#nn.GELU(),#nn.LeakyReLU(),#nn.GELU(),#nn.ReLU(),
            )
            self.embedding_mue = nn.Sequential(
            nn.Linear(self.d_model,self.hidden),
            nn.ELU(),#nn.GELU(),#nn.LeakyReLU(), #nn.GELU(),#nn.ReLU(),
            nn.Linear(self.hidden,self.hidden//2),
            nn.ELU(),
            nn.Linear(self.hidden//2,self.hidden//4),
            nn.ELU()#nn.GELU(),#nn.LeakyReLU(),#nn.GELU(),#nn.ReLU(),
            )

            self.pis = nn.Sequential(
            nn.Linear(self.d_model,self.hidden),
            nn.ELU(),
            nn.Linear(self.hidden,self.hidden//2),
            nn.ELU(),#nn.GELU(),#nn.LeakyReLU(), #nn.GELU(),#nn.ReLU(),
            nn.Linear(self.hidden//2,self.hidden//4),
            nn.ELU(),
            nn.Linear(self.hidden//4,self.gaussians)
            #nn.Softmax()
            )

            
            self.hidden_hid = self.hidden//4
            # self.pis = nn.Linear(self.hidden_hid,self.gaussians).to(device)
            self.sigma_x = nn.Linear(self.hidden_hid, self.gaussians)
            self.sigma_y = nn.Linear(self.hidden_hid, self.gaussians)
            self.mu_x = nn.Linear(self.hidden_hid, self.gaussians)
            self.mu_y = nn.Linear(self.hidden_hid, self.gaussians)

    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask: torch.Tensor=None, 
                tgt_mask: torch.Tensor=None) -> torch.Tensor:
        # Embedding 
        encoder_embed = self.encoder_input_layer(src).to(device)
        encoder_embed = self.positional_encoding_layer(encoder_embed).to(device) 

        # src shape: [batch_size, enc_seq_len, dim_val]
        encoder_out = self.encoder(src=encoder_embed).to(device)

        # Pass decoder input through decoder input layer
        decoder_embed = self.decoder_input_layer(tgt).to(device)
        decoder_output = self.positional_encoding_layer(decoder_embed).to(device) 

        # Pass throguh decoder - output shape: [batch_size, target seq len, dim_val]
        decoder_output = self.decoder(
            tgt=decoder_output,
            memory=encoder_out,
            tgt_mask=tgt_mask,
            memory_mask=src_mask
            ).to(device)

        #mdn_embeded = self.embedding_mdn(decoder_output).to(device)
        sigmax_embeded = self.embedding_sigma(decoder_output).to(device)
        sigmay_embeded = self.embedding_sigma(decoder_output).to(device)
        muex_embeded = self.embedding_mue(decoder_output).to(device)
        muey_embeded = self.embedding_mue(decoder_output).to(device)

        # Calculate PI
        pi = self.pis(decoder_output).to(device)
        pi = nn.functional.softmax(pi, -1)

        # Calculate Sigmas
        sigma_x = torch.Tensor(torch.exp(self.sigma_x(sigmax_embeded))).to(device)
        sigma_y = torch.Tensor(torch.exp(self.sigma_x(sigmay_embeded))).to(device)

        mu_x = torch.Tensor(self.mu_x(muex_embeded)).to(device)
        mu_y = torch.Tensor(self.mu_y(muey_embeded)).to(device)
       
        #result = torch.Tensor(decoder_output)

        return  pi, sigma_x,sigma_y, mu_x ,mu_y,decoder_output
  
if __name__ == "__main__":
     # Train transformer only
    in_features = 2
    out_features = 3
    num_heads = CFG.num_heads
    num_encoder_layers = CFG.num_encoder_layers
    num_decoder_layers =  CFG.num_decoder_layers
    embedding_size = CFG.embd_size
    max_length = 8
    n_hidden = 10
    gaussians = 5
    forecast_window = 12
    drp=0.2
    # transformer_mdn = Transformer_MDN( 
    #         device,
    #         in_features,
    #         out_features,
    #         num_heads,
    #         num_encoder_layers,
    #         num_decoder_layers,
    #         embedding_size,
    #     ).to (device)
    attn_mdn = Attention_GMM_Encoder(device,in_features,out_features,num_heads,num_encoder_layers,num_decoder_layers,embedding_size,n_gaussians=gaussians,n_hidden = n_hidden, dropout=drp).to(device)

    #attn_mdn = Attention_GMM(device,in_features,out_features,num_heads,num_encoder_layers,num_decoder_layers,embedding_size).to(device)
    #print(transformer_mdn)
    for name, child in attn_mdn.named_children():
        print(name, child)
    inp = torch.rand(16,7,2).to(device)
    tgt = torch.rand(16,12,3).to(device)
    tgt_mask = generate_square_mask(dim_trg = 12 ,dim_src = 12, mask_type="tgt").to(device)


    pi, sigma_x,sigma_y, mu_x , mu_y,decoder_out = attn_mdn(inp,tgt,tgt_mask = tgt_mask)
    print('pi shape',pi.shape,'\nsigma_x shape:',sigma_x.shape,'\nmu_x shape: ',mu_x.shape)