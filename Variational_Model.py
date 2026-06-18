from torch import nn
import torch
import numpy as np
import math
import copy

def gaussian_init_(n_units, std=1):    
    sampler = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([std/n_units]))
    Omega = sampler.sample((n_units, n_units))[..., 0]  
    return Omega

### Default Values ###
#dropHidden=0.35     #
#dropKoop=0.10       #
#ActFunc = nn.Tanh() # 
######################

class dynamics(nn.Module):
    def __init__(self, b,bx, init_scale,dropKoop=0.10):
        super(dynamics, self).__init__()
        
        self.dynamics = nn.Linear(b, b, bias=False)
        self.dropout = nn.Dropout(dropKoop)
        self.hiddenlayers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.bx=bx

        for j in range(self.bx):
            self.hiddenlayers.append( nn.Linear(hidden_dim, hidden_dim) )
            self.dropouts.append( nn.Dropout(dropKoop) )
        
        self.dynamics.weight.data = gaussian_init_(b, std=1)           
        U, _, V = torch.svd(self.dynamics.weight.data)
        self.dynamics.weight.data = torch.mm(U, V.t()) * init_scale

        
    def forward(self, x):
        x = self.dynamics(x)
        x = self.dropout(x)
        for j in range(self.bx):
            ## COM OU SEM self.activation(self.hiddenlayers[j](x)) ???
            x = self.hiddenlayers[j](x)
            x = self.dropouts[j](x)
        
        return x

class dynamics_back(nn.Module):
    def __init__(self, b,bx, omega,dropKoop=0.10):
        super(dynamics_back, self).__init__()
        self.dynamics = nn.Linear(b, b, bias=False)
        self.dropout = nn.Dropout(dropKoop)
        self.dynamics.weight.data = torch.pinverse(omega.dynamics.weight.data.t())  

        self.hiddenlayers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.bx=bx

        for j in range(self.bx):
            self.hiddenlayers.append( nn.Linear(hidden_dim, hidden_dim) )
            self.dropouts.append( nn.Dropout(dropKoop) )
            

    def forward(self, x):
        x = self.dynamics(x)
        x = self.dropout(x)
        for j in range(self.bx):
            ## COM OU SEM self.activation(self.hiddenlayers[j](x)) ???
            x = self.hiddenlayers[j](x)
            x = self.dropouts[j](x)
        return x

class BetaNet(nn.Module):
    ## Código que paraleliza o AB3
    def __init__(self, dim, n, Bound_dim,numet=None,type=None): ## levels é o número de camadas intermediarias
        super(BetaNet, self).__init__()
        #dim é a dimensao do sistema
        # bound dim é a dimensao das fronteiras (3a ordem espacial)
        # n é a dimensao temporal (3a ordem: presente, passado, passado do passado)

        self.typeA = (type =='A')
        self.typeB = (type =='B')
        self.input_dim = (dim + Bound_dim) * n
        print('BetaNet. Dimensao do vetor de entrada na rede = ', self.input_dim)
        Dx, Dt, f, g = (1, 1, 1, 1) if numet is None else (numet.dom.dx, numet.dom.dt, numet.sw.f, numet.sw.g)
        CFL = Dt/Dx

        self.beta = nn.Parameter(torch.tensor(0.281105))
        self.encoder = [VencoderNet(None,None,None,None,type='D')]

        if self.typeA: # momentum
            bilinear_indim1 = 8 # Drhs
            bilinear_indim2 = 7 # U.V
            bilinear_outdim = 7 # uh, vh
           
            self.bilinear = nn.Bilinear(bilinear_indim1, bilinear_indim2, bilinear_outdim, bias=False)
            
            hs = [[2,0],[2,1],[5,0],[2,3],
                  [2,4],[0,6],[0,7]]
            with torch.no_grad():
             self.bilinear.weight.fill_(0)
             for n, (a, b) in enumerate(hs):
             # [índice_saída, índice_v1, índice_v2]
              self.bilinear.weight[n, a, n] = 0.5
              self.bilinear.weight[n, b, n] = 0.5

            self.bilinear2 = nn.Bilinear(13, 11, 1, bias=False)
            with torch.no_grad():
             self.bilinear2.weight.fill_(0)
                ### 0
             self.bilinear2.weight[0,0,0] = g*CFL/2
             self.bilinear2.weight[0,1,1] = -g*CFL/2
             self.bilinear2.weight[0,0,2] = g*CFL/2
             self.bilinear2.weight[0,0,3] = g*CFL/2
             self.bilinear2.weight[0,1,2] = -g*CFL/2
             self.bilinear2.weight[0,1,3] = -g*CFL/2
                ### 1             
             self.bilinear2.weight[0, 3:9, 4] = CFL/ 4 # alguns sao negativos
             self.bilinear2.weight[0, 2, 5] = CFL / 4
             self.bilinear2.weight[0, 2, 6] = - CFL/ 4 # negativo
             self.bilinear2.weight[0, 3, 5] = CFL / 4
             self.bilinear2.weight[0, 4, 6] = CFL / 4 # negativo
             self.bilinear2.weight[0, 5, 7] = CFL / 4 # negativo
             self.bilinear2.weight[0, 7, 7] = CFL / 4 # negativo
             self.bilinear2.weight[0, 6, 8] = CFL / 4
             self.bilinear2.weight[0, 8, 8] = CFL / 4
             for n in [4,5,7]:
                 self.bilinear2.weight[0,n,4:9]*=-1
                 #### 2
             self.bilinear2.weight[0, 9, 9] = f*CFL/4
             self.bilinear2.weight[0,10, 9] = f*CFL/4
             self.bilinear2.weight[0,11,10] = f*CFL/4
             self.bilinear2.weight[0,12,10] = f*CFL/4

             self.bilinear2.weight.requires_grad = False
                   
        else: # zeta
            bilinear_indim1 = 5 # Drhs
            bilinear_indim2 = 4 # momentum
            bilinear_outdim = 1 # momentum
            self.bilinear = nn.Bilinear(bilinear_indim1 , bilinear_indim2, bilinear_outdim, bias=False) # bilinear para multiplicoçoes de uh e vh
            with torch.no_grad():
             self.bilinear.weight.fill_(0)
             # [índice_saída, índice_v1, índice_v2]
             self.bilinear.weight[0, 0, 0] = CFL/2 #h0*u[0]
             self.bilinear.weight[0, 2, :] = CFL/2 #h2*u[:]
             self.bilinear.weight[0, 1, 1] = CFL/2
             self.bilinear.weight[0, 3, 2] = CFL/2
             self.bilinear.weight[0, 4, 3] = CFL/2
             for n in [0,2]: # Termos em u0 e u2 contribuem negativamente
                 self.bilinear.weight[0,:,n]*=-1
             
       #self.beta.requires_grad = False
       #self.bilinear.weight.requires_grad = False
       # Non lienar activation layers
       #self.FC_input1 = nn.Linear(6, self.hidden_dim, bias=False)
       #self.dropout1 = nn.Dropout(dropHidden)
       #self.FC_input2 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
       #self.dropout2 = nn.Dropout(dropHidden)
       #self.FC_out  = nn.Linear(self.hidden_dim, latent_dim, bias=False)
        
    def forward(self, x, n_variables, mode='forward',steps=1,bulk_input = None,delta=0):
        
        x = x.view(-1, self.input_dim)
        if self.typeA:
            # Reorganiza o vetor achatado em [Batch, 5 vizinhos, 4 variáveis por vizinho]
            x_h    = x[:, 0:32].view(-1, 8, 4)     #(j+1,j-1,j,i+1,i-1,j+2,i+1ej+1,i-1ej+1) - z_{t,t-1,t-2} e topo
            zeta   = x[:, 32:34].view(-1, 2)    #(ij,j+1)
            x_u    = x[:, 34:49].view(-1, 5, 3)    #(ij,j-1,j+1,i+1,i-1)                     - u_{t,t-1,t-2}
            x_v    = x[:, 49:61].view(-1, 4, 3)    #(ij,i-1,j+1,i-1ej+1)                     - v_{t,t-1,t-2}
            Dummy,f,g = x[:, -6].view(-1, 1),x[:, -5].view(-1, 1),x[:, -4].view(-1, 1)
            dx = x[:, -3].view(-1, 1)
            dy,dt  = x[:, -2].view(-1, 1),x[:, -1].view(-1, 1)
            cf0  = 0.614
            cf2  = 0.088 
            cf1  = 0.285
            cf3  = 0.013
            h2   = x_h[:, :, 0]  # Todas as h2 de todos os vizinhos (j+1,j-1,j,i+1,i-1,j+2,i+1ej+1,i-1ej+1)
            h1   = x_h[:, :, 1]  # Todas as h1
            h0   = x_h[:, :, 2]  # Todas as h0
            topo = x_h[:, :, 3] # Todos os topos
            u2   = x_u[:, :, 0]  # Todas as u2 de todos os vizinhos (ij,j-1,j+1,i+1,i-1)
            u1   = x_u[:, :, 1]  # Todas as u1
            u0   = x_u[:, :, 2]  # Todas as u0
            v2   = x_v[:, :, 0]  # Todas as v2 de todos os vizinhos (ij,i-1,j+1,i-1ej+1)
            v1   = x_v[:, :, 1]  # Todas as v1
            v0   = x_v[:, :, 2]  # Todas as v0
            x_top   = torch.stack((topo[:, 2],topo[:, 0]),dim=1)     #(ij,j+1)
            x_zeta2 = torch.stack((h2[:, 2],h2[:, 0]),dim=1)     #(ij,j+1)
            x_zeta1 = torch.stack((h1[:, 2],h1[:, 0]),dim=1)     #(ij,j+1)
            x_zeta0 = torch.stack((h0[:, 2],h0[:, 0]),dim=1)     #(ij,j+1)
            
            h = h2 * (1.5 + self.beta) - (0.5 + 2 * self.beta) * h1 + self.beta * h0 + topo
            z_new = zeta*cf0 + x_zeta2 * cf1 + x_zeta1*cf2 + x_zeta0*cf3
            u = u2 * (1.5 + self.beta) - (0.5 + 2 * self.beta) * u1 + self.beta * u0
            v = v2 * (1.5 + self.beta) - (0.5 + 2 * self.beta) * v1 + self.beta * v0
            uv   = torch.concat((u[:,:3],v),dim=1) # quero apenas u(ij,j-1,j+1)

            h_ = self.bilinear(h,uv) 
            T1 = torch.concat((z_new,h_,v),dim=1)
            T2 = torch.concat((z_new,x_top,u,h[:,2:3]*dx,h[:,0:1]*dx),dim=1)
            U_ = self.bilinear2(T1,T2)

            return [U_],None,None,None,None,None,None
            
        elif self.typeB:
        # Reorganiza o vetor achatado em [Batch, 5 vizinhos, 4 variáveis por vizinho]
            x_h = x[:, 0:20].view(-1, 5, 4)  #(j+1,j-1,ij,i+1,i-1) - z_{t,t-1,t-2} e topo
            x_u = x[:, 20:26].view(-1, 2, 3) #(ij, j-1)                - u_{t,t-1,t-2}
            x_v = x[:, 26:32].view(-1, 2, 3) #(ij i-1)                - v_{t,t-1,t-2}
            #dx,dy,dt = x[:, -3].view(-1, 1),x[:, -2].view(-1, 1),x[:, -1].view(-1, 1)

            h2 = x_h[:, :, 0]  # Todas as h2 de todos os vizinhos (j+1,j-1,ij,i+1,i-1)
            h1 = x_h[:, :, 1]  # Todas as h1
            h0 = x_h[:, :, 2]  # Todas as h0
            topo = x_h[:, :, 3] # Todos os topos
            h = h2 * (1.5 + self.beta) - (0.5 + 2 * self.beta) * h1 + self.beta * h0 + topo
            u2 = x_u[:, :, 0]  # Todas as u2 de todos os vizinhos (ij, j-1)
            u1 = x_u[:, :, 1]  # Todas as u1
            u0 = x_u[:, :, 2]  # Todas as u0
            u = u2 * (1.5 + self.beta) - (0.5 + 2 * self.beta) * u1 + self.beta * u0
            v2 = x_v[:, :, 0]  # Todas as v2 de todos os vizinhos (ij i-1)
            v1 = x_v[:, :, 1]  # Todas as v1
            v0 = x_v[:, :, 2]  # Todas as v0
            v = v2 * (1.5 + self.beta) - (0.5 + 2 * self.beta) * v1 + self.beta * v0
#
            uv   = torch.concat((u,v),dim=1)
            H_ = self.bilinear(h,uv)
            
            return [H_],None,None,None,None,None,None
    
    
class VencoderNet(nn.Module):
    def __init__(self, dim, n, Bound_dim, latent_dim, hidden_dim = 1, ALPHAx = 1,
                 ActFunc = nn.Tanh(), dropHidden=0.35,type='C',numet=None): ## levels é o número de camadas intermediarias
        super(VencoderNet, self).__init__()
        if type == 'D':
            self.typeC = False
            return
        #dim é a dimensao do sistema
        # bound dim é a dimensao das fronteiras (3a ordem espacial)
        # n é a dimensao temporal (3a ordem: presente, passado, passado do passado)
        # extra_dim são o dx e dt, inputs sobre a resolucao expacial e temporal do modelo.
        
        self.input_dim = (dim + Bound_dim) * n
        print('Dimensao do vetor de entrada na rede = ', self.input_dim)
        self.hidden_dim = 16*hidden_dim
        self.activation = ActFunc
        self.hiddenlayers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.ALPHAx=ALPHAx
        self.typeAB = ('A' == type or 'B' == type) # Momentum_pinn Encoder
        self.typeC = ('C' == type) # Trivial Encoder
        beta = 0.281105
            
        if self.typeAB:

            self.pinn_encoder = BetaNet(dim, n, Bound_dim,numet=numet,type=type)
            
            self.bn0 = nn.BatchNorm1d(6 ) #, affine=False
            self.FC_input = nn.Linear(6, self.hidden_dim) #, bias=False
            #self.bn0 = nn.BatchNorm1d(self.input_dim-5 ) #, affine=False
            #self.FC_input = nn.Linear(self.input_dim-5, self.hidden_dim) #, bias=False
            
        elif self.typeC:
            self.bn0 = nn.BatchNorm1d(self.input_dim) #, affine=False
            self.FC_input = nn.Linear(self.input_dim, self.hidden_dim) #, bias=False
            
        self.dropout1 = nn.Dropout(dropHidden)
        for j in range(self.ALPHAx):
            self.hiddenlayers.append( nn.Linear(self.hidden_dim, self.hidden_dim) ) #, bias=False
            self.dropouts.append( nn.Dropout(dropHidden) )

        # ENCODER EXIT
        self.FC_input2 = nn.Linear(self.hidden_dim, self.hidden_dim) #, bias=False
        self.dropout2 = nn.Dropout(dropHidden)
        self.FC_mean  = nn.Linear(self.hidden_dim, latent_dim)
        self.FC_var   = nn.Linear (self.hidden_dim, latent_dim)
        
            
    def forward(self, x):
        
        # SINCE THE AE USES LINEAR LAYERS AND NO CONVS       #
        # IT CAN BE A 2D VECTOR WITH SHAPE [BATCH,INPUT_DIM] #
        x = x.view(-1, self.input_dim)
        x_ = None
        dx,dy,dt = x[:, -3].view(-1, 1),x[:, -2].view(-1, 1),x[:, -1].view(-1, 1)
        
        if self.typeAB:
        
            h_ = self.pinn_encoder(x,n_variables=1)[0][0].view(-1,1)

            h_ = torch.concat((h_,dy,1/dy,dy, # seria dx,1/dx... mas dx em V esta negativo por causa do betaNet.
                               1/dy,dt),dim=1)

            #h_ = torch.concat((x[:,:-6],h_),dim=1)
                
            x = self.bn0(h_)

            #print('TypeAB, SAIDA bn0:',x[:,0].view(200,200)[50:55,50:55].tolist(),
            #      #x[:,1].view(200,200)[50:55,50:55],x[:,2].view(200,200)[50:55,50:55],
            #      #x[:,3].view(200,200)[50:55,50:55],x[:,4].view(200,200)[50:55,50:55]
            #)
        elif self.typeC:
            # Normalizes Input inside network #
            ############## # constant network resolution for testing # ###############
            #batch_size = x.shape[0]                                                   #
            #dx     = torch.full((batch_size, 1), 200157.7794640436, device=x.device)  #
            #inv_dx = 1 / dx                                                           #
            #dt     = torch.full((batch_size, 1), 360.0, device=x.device)              #
            #res_tensor = torch.cat([dx, inv_dx, dx, inv_dx, dt], dim=1)               #
            #print(res_tensor.shape)                                                   #
            #x[:,1:] = res_tensor                                                      #
            #############################(comment during training )###################
            
            x = self.bn0(x)
            #print('TypeC, SAIDA bn0:',x[:,0].view(200,200)[50:55,50:55].tolist(),
            #      #x[:,1].view(200,200)[50:55,50:55],x[:,2].view(200,200)[50:55,50:55],
            #      #x[:,3].view(200,200)[50:55,50:55],x[:,4].view(200,200)[50:55,50:55]
            #)
        # Non-linear Activation Encoder #
        x = self.FC_input(x)
        x = self.activation(x)
        h_ = self.dropout1(x)
        #hidden layers
        for j in range(self.ALPHAx):
            h_ = self.activation(self.hiddenlayers[j](h_))
            h_ = self.dropouts[j](h_)
        h_ = self.FC_input2(h_)
        h_ = self.activation(h_)
        h_ = self.dropout2(h_)
        mean     = self.FC_mean(h_)
        log_var  = self.FC_var(h_)                     # encoder produces mean and log of variance 
                                                       #             (i.e., parateters of simple tractable normal distribution "q"
        return mean, log_var, x_
    
class VdecoderNet(nn.Module):
    def __init__(self, output_dim, input_dim, hidden_dim = 1, BETAx = 1,
                 ActFunc = nn.Tanh(),dropHidden=0.35):
        super(VdecoderNet, self).__init__()

        self.activation = ActFunc
        self.hidden_dim = 16*hidden_dim
        self.input_dim  = input_dim
        self.output_dim = output_dim
        self.hiddenlayers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.BETAx=BETAx
        
        self.FC_hidden = nn.Linear(self.input_dim, self.hidden_dim, bias=False)
        self.dropout1 = nn.Dropout(dropHidden)
        #self.ln1 = nn.LayerNorm(self.hidden_dim)
        
        for j in range(self.BETAx):
            self.hiddenlayers.append( nn.Linear(self.hidden_dim, self.hidden_dim, bias=False) )
            self.dropouts.append( nn.Dropout(dropHidden) )
        self.FC_hidden2 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.dropout2 = nn.Dropout(dropHidden)
        self.FC_output = nn.Linear(self.hidden_dim, self.output_dim, bias=False)

    def forward(self, x):
        
        # SINCE THE AE USES LINEAR LAYERS AND NO CONVS       #
        # IT CAN BE A 2D VECTOR WITH SHAPE [BATCH,INPUT_DIM] #
        x = x.view(-1, self.input_dim)
        h = self.FC_hidden(x)
        #   Layer NORM  #
        #x = self.ln1(x)#
        #################
        h = self.activation(h)
        h = self.dropout1(h)
        
        for j in range(self.BETAx):
            h = self.activation(self.hiddenlayers[j](h))
            h = self.dropouts[j](h)
        h = self.FC_hidden2(h)
        h = self.activation(h)
        h = self.dropout2(h)
        
        x_hat = self.FC_output(h)
        x_hat = x_hat.view(-1, 1, self.output_dim, 1)
        
        return x_hat
    
class VLieAE(nn.Module):
    def __init__(self, dim, dim_out, n, Bound_dim, b, n_variables=1, alpha = 1, beta = 1,
                 alphax = 0, betax = 0, bx = 0, init_scale=1, multiencoder=False, multidynamics = False, ActFunc = nn.Tanh(),
                 dropKoop=0.10,dropHidden=0.35,type='D',numet=None):
        
        ## Multi-variable Variational Lie AUTOENCODER ##
        ## ## ### ### ### ### ### ### ### ### ### ### ### ### ### ## ## #### ### ### #### ### ### ## ## #### ### #
        ## dim é a dimensão do sistema input                                                                    ##
        ## dim_out é uma lista com as dimensões de saída de cada uma das variaveis que a rede vai evoluir        #
        ## n é a dimensão de cores ( =1)                                                                         #
        ## Bound_dim é a dimnesão dos boundaries (Util??)                                                       ##
        ## b é o bottleneck, dimensão do operador do lattent space (Koopman)                                     #
        ## alphax, betax e bx são o numero de camadas ocultas em cada parte da rede                              #
        ## Bulk_dim>0 - Utiliza o Controldecoder: forward da rede + input de bulk (control) para gerar saída     #
        ## AE dim substitui o n_variables:
        ## diferentes AEs para que diferentes input-vectors vao p/ o mesmo latent space
        ## type 'A' é a rede com uma camada linear e batch norm no encoder. type 'B' tem encoder decoder simetricos #
        ## ## ### ### ### ### ### ### ### ### ### ### ### ### ## ## #### ### ### ### ### ## ## #### ### ### #### ####

        self.n_varaibles = n_variables
        
        super().__init__()
        self.multiencoder = multiencoder
        self.multidynamics = multidynamics
        
        self.encoder        = nn.ModuleList()
        self.decoder        = nn.ModuleList()
        self.Controldecoder = nn.ModuleList()
        self.n_variables    = n_variables
        self.std_max        = None

        ## Antigo, para Control Theory do Bulk.
        self.Bulk_dim=0
        #self.Bulk_dim     = Bulk_dimVencoderNet(dim, n, Bound_dim, b, ALPHA = alpha, ALPHAx = alphax,ActFunc=ActFunc,dropHidden=dropHidden)

        self.dynamics     = nn.ModuleList()
        self.backdynamics = nn.ModuleList()
        if not multidynamics:
            self.dynamics.append(dynamics(b,bx, init_scale,dropKoop=dropKoop))
            self.backdynamics.append(dynamics_back(b,bx, self.dynamics[0],dropKoop=dropKoop))
        if not multiencoder:
            self.encoder.append(VencoderNet(dim, n, Bound_dim, b, hidden_dim = alpha, ALPHAx = alphax,
                                            ActFunc=ActFunc,dropHidden=dropHidden,type=type,numet=numet))
        
        for i in range(n_variables): # CADA VARIAVEL TEM UMA DINAMICA E DECODER NO KOOPMAN SPACE
            if multidynamics:
                self.dynamics.append(dynamics(b,bx, init_scale,dropKoop=dropKoop))
                self.backdynamics.append(dynamics_back(b,bx, self.dynamics[i],dropKoop=dropKoop))
            if multiencoder:
                self.encoder.append(VencoderNet(dim, n, Bound_dim, b, hidden_dim = alpha,ALPHAx = alphax,
                                                ActFunc=ActFunc,dropHidden=dropHidden,type=type,numet=numet))
            self.decoder.append(VdecoderNet(dim_out[i], b, hidden_dim = beta, BETAx = betax, 
                                            ActFunc = ActFunc,dropHidden=dropHidden))
            if self.Bulk_dim:
                self.Controldecoder.append(VdecoderControlNet(dim_out[i], b, hidden_dim = beta, BETAx = betax,ActFunc = ActFunc,dropHidden=dropHidden))

    def reparameterization(self, mean, var, delta):

        epsilon = torch.randn_like(var)*delta          # sampling epsilon from a normal distribution of the same dimensions as var...       
        z = mean + var*epsilon                          # reparameterization trick
        
        return z

    def forward(self, x, n_variables, mode='forward',steps=1,bulk_input = None,delta=1):
        ## MODIFICADO PARA USAR multiencoder e ou multidynamics.
        out           = []
        out_back      = []
        out_ae        = []

        if self.multiencoder:
            Encoded = []
            mean, log_var = [],[]
    
            if len(x) != len(self.encoder):
                print('Numero distinto de input vectors (', len(x),') e AE.')
                return 'deu ruim'
            
            for i in range(len(x)):
                m, l_v,x_ = self.encoder[i](x[i].contiguous())
                mean.append(m)
                log_var.append(l_v)
        
                # Aqui seria possível fazer a evolução de um média do vetor sorteado da distribuição gaussina n dimensional.
                Encoded.append(self.reparameterization(m, torch.exp(0.5 * l_v),delta) ) # takes exponential function (log var -> var)
        else:
            # Há apenas um Encoder na lista.
            mean, log_var, x_ = self.encoder[0](x.contiguous())
            # Aqui seria possível fazer a evolução de um média do vetor sorteado da distribuição gaussina n dimensional.
            Encoded = self.reparameterization(mean, torch.exp(0.5 * log_var),delta)  # takes exponential function (log var -> var)
            
        if mode == 'forward':
            for _ in range(steps): #self.steps-2
                
                ## Antigo para multiplas variaveis   
                for n in range(n_variables):
                    if self.multiencoder:
                        z = Encoded[n].contiguous()
                    else:
                        z = Encoded.contiguous()
                    out_ae.append(self.decoder[n](z))
                    if self.multidynamics:
                        Forwarded = self.dynamics[n](z) # Testando 1 dinamica para diferentes encoders
                    else:
                        Forwarded = self.dynamics[0](z)
                    
                    if self.Bulk_dim>0:  # CONTROL THEORY #           
                        # une Forwarded e bulk_input do presente e passa pelo self.Controldecoder
                        Control_vector = torch.concatenate([Forwarded[-1],bulk_input],axis=2)
                        out.append(self.decoder[n](Forwarded))
                    else:
                        out.append(self.decoder[n](Forwarded)) 
            # saída da 1a camada do encoder
            #if self.typeA:
            out.append(x_)
            return out, out_back, out_ae, Encoded, Forwarded, mean, log_var

        elif mode == 'backward':
            for _ in range(steps): #self.steps-2
         
                ## Antigo para multiplas variaveis    
                for n in range(n_variables):
                    if self.multiencoder:
                        z = Encoded[n].contiguous()
                    else:
                        z = Encoded.contiguous()
                    
                    out_ae.append(self.decoder[n](z))
                    if self.multidynamics:
                        Forwarded = self.backdynamics[n](z)
                    else:
                        Forwarded = self.backdynamics[0](z)

                    if self.Bulk_dim>0:
                        # une Forwarded e bulk_input do passo anterior e passa pelo self.Controldecoder
                        Control_vector = torch.concatenate([Forwarded,bulk_input],axis=2)
                        out_back.append(self.decoder[n](Forwarded))
                    else:
                        out_back.append(self.decoder[n](Forwarded))
            # saída da 1a camada do encoder
            #if self.typeA:
            #   out_back.append(x_)           
            return out, out_back, out_ae, Encoded, Forwarded, mean, log_var

        elif mode == 'encode':
            for _ in range(steps): #self.steps-2
                
                ## Antigo para multiplas variaveis
                for n in range(n_variables):
                    if self.multiencoder:
                        z = Encoded[n].contiguous()
                    else:
                        z = Encoded.contiguous()
                    out_ae.append(self.decoder[n](z))
                    
            Forwarded = []
            # saída da 1a camada do encoder
            #out_ae.append(x_)        
            return out, out_back, out_ae, Encoded, Forwarded, mean, log_var

    def Eigen_functions(self,mode = 'forward'):

        if mode == 'forward': 
            eigenvalues, eigenvectors = torch.linalg.eig(self.dynamics[0].dynamics.weight)
        elif mode == 'backward': 
            eigenvalues, eigenvectors = torch.linalg.eig(self.backdynamics[0].dynamics.weight)

        # Calculate the norm (magnitude) of the eigenvalues
        norms = torch.abs(eigenvalues)
        
        # Get the sorted indices based on the norms
        sorted_indices,sorted_norm = torch.sort(norms)[::-1]  # Sort by magnitude, descending

        # Sort eigenvalues and eigenvectors by the norms
        sorted_eigenvalues = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]

        self.eigenvalues,self.eigenvectors = sorted_eigenvalues, sorted_eigenvectors
        
        return eigenvalues, eigenvectors

    ###########
    ### DMD ###
    ########### 

class DMD(nn.Module):
    def __init__(self,n_variables):
        super(DMD, self).__init__()
        self.dynamics = None  # You can initialize your A matrix here if needed
        self.n_variables = n_variables

    def forward(self, X, n_variables, mode='forward',steps=1,train=False,delta=None):
        # X com os input vectors empilhados (input_dim,espaço*tempo)
        # Faz a previsão baseado na dinamica salva do DMD
        # a saida out tem a dimensao espaco e tempo empilhada Y_pred.shape = (1,espaço*tempo)
        # se estiver evoluindo apenas 1 passo no tempo, não tem problema...
        device = torch.device("cpu")
        out = []
        out_back = []
        out_ae = []
        mean, log_var = [],[]
        Encoded = []
        Forwarded = []
        if mode == 'forward':
            out.append(self.dynamics.to(device) @ X.to(device))  # shape (1, 100000)
        else:
            out_back.append(self.dynamics.to(device) @ X.to(device))  # shape (1, 100000)
        
        return out, out_back, out_ae, Encoded, Forwarded, mean, log_var


    def train(self, X, Y):
        # dado uma matriz de 2 dimensoes X com os input vectors empilhados (input_dim,espaço*tempo) e Y de solucoes empilhadas (1,espaço*tempo),
        # A é o vetor que transforma AX=Y
        
        A = Y @ np.linalg.pinv(X)  # shape (1, dim(X))
        A, _, _, _ = np.linalg.lstsq(X.T, Y.T, rcond=None)
        A = torch.from_numpy(A)
        self.dynamics = A.T  # shape (1, dim(X))

                   ##########
    ### Feed Fwd Network ####
    ########### 

class MultilayerNetwork(nn.Module):
    def __init__(self, layers, dim, n, Bound_dim, ActFunc, dropHidden, type='C',numet=None):

        # Rede multilayer - com função de ativação.

        super(MultilayerNetwork, self).__init__()
        
        self.input_dim = (dim + Bound_dim) * n
        print('dimensao do vetor de entrada na rede = ', self.input_dim)
        print('Dropout = ', dropHidden)
        self.activation = ActFunc
        self.hiddenlayers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.typeAB = (type == 'A' or type == 'B')
        self.typeC  = (type == 'C')
        
        if self.typeAB:
            self.pinn_encoder = BetaNet(dim, n, Bound_dim,numet=numet,type=type)
    
            #for i in range(n_variables): # CADA VARIAVEL TEM UMA DINAMICA E DECODER NO KOOPMAN SPACE
            self.FC_input = nn.Linear(6, layers[0])
            self.bn0 = nn.BatchNorm1d(6)
        else:
            self.bn0 = nn.BatchNorm1d(self.input_dim)
            self.FC_input = nn.Linear(self.input_dim, layers[0])
        self.dropout1 = nn.Dropout(dropHidden)
        for j in range(1,len(layers)):
            self.hiddenlayers.append( nn.Linear(layers[j-1], layers[j]) )
            self.dropouts.append( nn.Dropout(dropHidden) )
        self.out   = nn.Linear (layers[-1], 1)
        
    def forward(self, x):
        x = x.view(-1, self.input_dim)
        dx,dy,dt = x[:, -3].view(-1, 1),x[:, -2].view(-1, 1),x[:, -1].view(-1, 1)
        
        if self.typeAB:
            
            x = self.pinn_encoder(x,n_variables=1)[0][0].view(-1,1)
            x = torch.concat((x,dy,1/dy,dy,1/dy,dt),dim=1) 
            
        x = self.bn0(x)    
        h_ = self.activation(self.FC_input(x))
        h_ = self.dropout1(h_)
        #hidden layers
        for j in range(len(self.hiddenlayers)):
            h_ = self.activation(self.hiddenlayers[j](h_))
            h_ = self.dropouts[j](h_)
        out = self.out(h_)

        return out


class Multilayers(nn.Module):
    def __init__(self, layers, dim, n, Bound_dim, dropHidden):

        # rede multicamadas linear - sem funcão de ativação.
        
        super(Multilayers, self).__init__()
        
        self.input_dim = (dim + Bound_dim) * n
        print('dimensao do vetor de entrada na rede = ', self.input_dim)
        print('Dropout = ', dropHidden)
        
        self.hiddenlayers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        #for i in range(n_variables): # CADA VARIAVEL TEM UMA DINAMICA E DECODER NO KOOPMAN SPACE
        self.FC_input = nn.Linear(self.input_dim, layers[0])
        self.dropout1 = nn.Dropout(dropHidden)
        for j in range(1,len(layers)):
            self.hiddenlayers.append( nn.Linear(layers[j-1], layers[j]) )
            self.dropouts.append( nn.Dropout(dropHidden) )
        self.out   = nn.Linear (layers[-1], 1)
        
    def forward(self, x):
        x = x.view(-1, 1, self.input_dim)
        
        h_ = self.FC_input(x)
        h_ = self.dropout1(h_)
        #hidden layers
        for j in range(len(self.hiddenlayers)):
            h_ = self.hiddenlayers[j](h_)
            h_ = self.dropouts[j](h_)
        out = self.out(h_)

        return out
    
class ML(nn.Module):
    def __init__(self, layers, dim, dim_out, n, Bound_dim, n_variables=1,
                 linear = False, ActFunc = nn.Tanh(), dropHidden=0.40,type='C',numet=None):

        super(ML, self).__init__()
        self.n_variables = n_variables
        self.encoder = [VencoderNet(None,None,None,None,type='D')]
        ## Multi Layer Network ##
        if not linear:
            self.network = MultilayerNetwork(layers, dim, n, Bound_dim, ActFunc, dropHidden,type,numet)
        else:
            self.network = Multilayers(layers, dim, n, Bound_dim, dropHidden)

    def forward(self, x, n_variables = None, mode=None,delta=None):
        # apenas para ser analogo a função da VKAE preciso
        # colocar as variaveis n_variables e mode.
        # Pela mesma razão retorno varias listas vazias.
        
        out = []
        out_back = []
        out_ae = []
        Encoded, Forwarded, mean, log_var = [], [], [], []

        z = x.contiguous()
        out.append(self.network(z))
        
        return out, out_back, out_ae, Encoded, Forwarded, mean, log_var

# --- Configuração Global --- #
class Args:
    def __init__(self):
        self.n_variables = 1 # Precisa ser definido nas variaveis do Simulador...
        self.wd = 0          # l2 penalty to weight size.
        self.lr = 0.001      # Learning rate
        self.lr_decay = 0.6  # Learning rate decay
        self.lamb = 1        # constante multiplicativa do loss_identity
        self.eta = 0         # constante multiplicativa do loss_consist (será atualizada)
        self.nu = 0.5        # constante multiplicativa do loss_bwd
        self.epochs = 0      # Número de épocas (será atualizado)
        self.folder = ''     # Pasta de saída (será atualizada)
        self.alpha = 0       # camada intermediária encoder
        self.alphax = 0      # numero de camadas ocultas no encoder
        self.beta = 0        # camada intermediária decoders
        self.betax = 0       # numero de camadas ocultas no decoder
        self.bottleneck = 0  # dimensão no espaço de Koopman
        self.bottleneckx = 0 # numero de camas ocultas no espaço de Koopman
        self.n = 1           # dimensão da "convolução" na rede
        self.init_scale = 1  # constante multiplicativa da svd inicial para setar o operador linear da rede
        self.temporal_res = 3
        self.dim = 0         # Sem Vetores do sistema no input, apenas acrescimos.
        self.dim_out = [1,1,1]
        self.Bdim = 0        # Boundaries dimension (será atualizada)
        self.NN = 'LieAE'    # Tipo de rede neural
