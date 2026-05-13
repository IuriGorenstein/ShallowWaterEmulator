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
    def __init__(self, dim, n, Bound_dim): ## levels é o número de camadas intermediarias
        super(BetaNet, self).__init__()
        #dim é a dimensao do sistema
        # bound dim é a dimensao das fronteiras (3a ordem espacial)
        # n é a dimensao temporal (3a ordem: presente, passado, passado do passado)

        self.input_dim = (dim + Bound_dim) * n
        print('dimensao do vetor de entrada na rede = ', self.input_dim)
        linear_indim = 5 # vetores h
        linear_udim = 2 # vetores u e v
        linear_outdim = 4 # h_bar
        bilinear_indim = 4 # momentum
        bilinear_outdim = 4 # momentum
         # Temporal means (rhs) #
        self.beta = nn.Parameter(torch.tensor(0.28))
        # Linear and bilinear for spatial means, Uh/Vh and spatial derivatives #
        self.linear = nn.Linear(linear_indim, linear_outdim, bias=False) # linear para médias espaciais de H
        #            h[:, 0]+h[:, 2]         h[:, 2]+h[:, 1]
        #            h[:, 3]+h[:, 2]         h[:, 2]+h[:, 4]
        hbars = [[0,2],[2,1],[3,2],[2,4]]
        with torch.no_grad():
         self.linear.weight.fill_(0) 
         for i, n in enumerate(hbars):      
          self.linear.weight[i, n] = 0.5    # Atribui 0.5 nas colunas especificadas em n
             
        self.bilinear = nn.Bilinear(linear_outdim, bilinear_indim, bilinear_outdim, bias=False) # bilinear para multiplicoçoes de uh e vh
        with torch.no_grad():
         self.bilinear.weight.fill_(0)
         for n in range(bilinear_outdim):
         # [índice_saída, índice_v1, índice_v2]
          self.bilinear.weight[n, n, n] = 1.0
            
        self.bilinear2 = nn.Bilinear(bilinear_outdim+1,5,1, bias=False) # bilinear para dividir por dx,dy,dt
        with torch.no_grad():
         self.bilinear2.weight.fill_(0)
         self.bilinear2.weight[0, 0,1] = -1.0# + (((h_p1 + h_0)/2)*u_0 )/ dx
         self.bilinear2.weight[0, 1,1] = 1.0 # - (((h_0 + h_m1)/2)*u_m1)/ dx
         self.bilinear2.weight[0, 2,3] = -1.0# + (((h_p1 + h_0)/2)*v_0 )/ dy
         self.bilinear2.weight[0, 3,3] = 1.0 # - (((h_0 + h_m1)/2)*v_m1)/ dy

       # Non lienar activation layers
       #self.FC_input1 = nn.Linear(6, self.hidden_dim, bias=False)
       #self.dropout1 = nn.Dropout(dropHidden)
       #self.FC_input2 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
       #self.dropout2 = nn.Dropout(dropHidden)
       #self.FC_out  = nn.Linear(self.hidden_dim, latent_dim, bias=False)
        
    def forward(self, x, n_variables, mode='forward',steps=1,bulk_input = None,delta=0):
        
        # SINCE THE AE USES LINEAR LAYERS AND NO CONVS       #
        # IT CAN BE A 2D VECTOR WITH SHAPE [BATCH,INPUT_DIM] #
        x = x.view(-1, self.input_dim)
        # REDE QUE TREINA UM PARAMETRO INICIAL PARA APROXIMAR O BETA, E EM SEGUIDA PASSA PARA O ENCODER...
        # Reorganiza o vetor achatado em [Batch, 5 vizinhos, 4 variáveis por vizinho]
        x_h = x[:, 0:20].view(-1, 5, 4)
        x_u = x[:, 20:26].view(-1, 2, 3)
        x_v = x[:, 26:32].view(-1, 2, 3)
        dx,dy,dt = x[:, -3],x[:, -2],x[:, -1]
        
        # Cria médias temporais rhs (aplicando parametro beta)
        h2 = x_h[:, :, 0]  # Todas as h2 de todos os vizinhos (ij, j-1, j+1, i-1, i+1)
        h1 = x_h[:, :, 1]  # Todas as h1
        h0 = x_h[:, :, 2]  # Todas as h0
        topo = x_h[:, :, 3] # Todos os topos
        h = h2 * (1.5 + self.beta) - (0.5 + 2 * self.beta) * h1 + self.beta * h0 + topo
        u2 = x_u[:, :, 0]  # Todas as u2 de todos os vizinhos (ij, j-1, j+1, i-1, i+1)
        u1 = x_u[:, :, 1]  # Todas as u1
        u0 = x_u[:, :, 2]  # Todas as u0
        u = u2 * (1.5 + self.beta) - (0.5 + 2 * self.beta) * u1 + self.beta * u0
        v2 = x_v[:, :, 0]  # Todas as v2 de todos os vizinhos (ij, j-1, j+1, i-1, i+1)
        v1 = x_v[:, :, 1]  # Todas as v1
        v0 = x_v[:, :, 2]  # Todas as v0
        v = v2 * (1.5 + self.beta) - (0.5 + 2 * self.beta) * v1 + self.beta * v0
                    
        uv   = torch.concat((u,v),dim=1)
        res  = torch.concat((dx.view(-1, 1),1/dx.view(-1, 1),dy.view(-1, 1),
                             1/dy.view(-1, 1),dt.view(-1, 1)),dim=1)
        hbar = self.linear(h)

        h_ = self.bilinear(hbar,uv) # uh e vh
        h_ = torch.concat((h_,torch.ones(hbar.shape[0], 1)),dim=1)
        h_ = self.bilinear2(h_,res)
        htend = h_*torch.squeeze(dt[0])
        
        return [htend],None,None,None,None,None,None
    
class VencoderNet(nn.Module):
    def __init__(self, dim, n, Bound_dim, latent_dim, hidden_dim = 1, ALPHAx = 1,
                 ActFunc = nn.Tanh(), dropHidden=0.35,type='D',Const=None,Dx=None): ## levels é o número de camadas intermediarias
        super(VencoderNet, self).__init__()
        #dim é a dimensao do sistema
        # bound dim é a dimensao das fronteiras (3a ordem espacial)
        # n é a dimensao temporal (3a ordem: presente, passado, passado do passado)
        # extra_dim são o dx e dt, inputs sobre a resolucao expacial e temporal do modelo.
        
        self.input_dim = (dim + Bound_dim) * n
        print('dimensao do vetor de entrada na rede = ', self.input_dim)
        self.hidden_dim = 16*hidden_dim
        self.activation = ActFunc
        self.hiddenlayers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.ALPHAx=ALPHAx
        self.typeA = ('A' == type) # Momentum_pinn Encoder
        self.typeB = ('B' == type) # Zeta_pinn Encoder
        self.typeC = ('C' == type) # Trivial Encoder
            
        if self.typeA:
            # Physics Informed Encoder for AM4 #
            # PARA ALIMENTAR OS pre_vetores, preciso de:
            # para -UFx(j)  : (-uh(i,j-1)-uh(i,j))/2*(-u(i,j-1)-u(i,j))/2       # v: (-vh(i-1,j)-vh(i,j))/2*(-v(i-1,j)-v(i,j))/2
            # para UFx(j+1) : (uh(i,j)+uh(i,j+1))/2*(u(i,j)+u(i,j+1))/2         # v: (vh(i-1,j+1)+vh(i,j+1))/2*(v(i-1,j+1)+v(i,j+1))/2
            # para UFe(i,j)   : (vh(i,j+1)+vh(i,j))/2*(u(i+1,j)+u(i,j))/2       # v: (vh(i+1,j)+vh(i,j))/2*(u(i,j+1)+u(i,j))/2 
            # para -UFe(i-1,j): (-vh(i-1,j+1)-vh(i-1,j))/2*(-u(i,j)-u(i-1,j))/2 # v: (-vh(i,j)-vh(i-1,j))/2*(-u(i-1,j+1)-u(i-1,j))/2 
            # Cor : h(i,j),h(i,j+1)                                             # v: h(i,j),h(i+1,j)
            #       v(i-1,j),v(i,j),v(i,j+1),v(i-1,j+1)                         # v: u(j-1,i),v(i,j),u(i+1,j),u(j-1,i+1)
            # GRAD :new_z(i,j),new_z(i,j+1),topo(i,j),topo(i,j+1)               # v: new_z(i,j),new_z(i+1,j),topo(i,j),topo(i+1,j)
            #       new_z(i,j),new_z(i,j+1),1                                   # v: new_z(i,j),new_z(i+1,j),1
            
            # No total: uh(i,j-1), uh(i,j), uh(i,j+1), vh(i,j+1), vh(i,j), vh(i-1,j+1), vh(i-1,j)
            #           u(i,j-1), u(i,j), u(i,j+1), u(i+1,j), u(i-1,j)
            #           v(i-1,j),v(i,j),v(i,j+1),v(i-1,j+1) 
            #           h(i,j),h(i,j+1)  
            #           new_z(i,j),new_z(i,j+1),topo(i,j),topo(i,j+1)
            
            # AGORA, para conseguir: uh(i,j-1), uh(i,j), uh(i,j+1), vh(i,j+1), vh(i,j), vh(i-1,j+1), vh(i-1,j).
            #           preciso dos: u(i,j-1) , u(i,j) , u(i,j+1) , v(i,j+1) , v(i,j) , v(i-1,j+1) , v(i-1,j).
            #                        h(i,j-1) ,  h(i,j), h(i,j+1) , h(i,j+1) , h(i,j) , h(i-1,j+1) , h(i-1,j).
            #                        h(i,j)  , h(i,j+1), h(i,j+2) ,h(i+1,j+1),h(i+1,j), h(i,j+1)   , h(i,j).
            #
            # LOGO, para U basta: u(i,j-1), u(i,j), u(i,j+1), u(i+1,j), u(i-1,j)
            #                     v(i-1,j),v(i,j),v(i,j+1),v(i-1,j+1) 
            #                     h(i,j),h(i,j+1),h(i,j+2),h(i+1,j+1),h(i+1,j),h(i-1,j+1),h(i,j-1),h(i-1,j)
            #                     new_z(i,j),new_z(i,j+1),topo(i,j),topo(i,j+1)
            
            linear_indim = 7 # vetores h
            linear_udim = 2 # vetores u e v
            linear_outdim = 6 # h_bar
            bilinear_indim = 6 # momentum
            bilinear_outdim = 6 # momentum
             # Temporal means (rhs) #
            self.beta = nn.Parameter(torch.tensor(0.28))
            # Linear and bilinear for spatial means, Uh/Vh and spatial derivatives #
            self.linear = nn.Linear(linear_indim, linear_outdim, bias=False) # linear para médias espaciais de H
            #            h[:, 0]+h[:, 2]         h[:, 2]+h[:, 1]
            #            h[:, 3]+h[:, 2]         h[:, 2]+h[:, 4]
            # vh(i+1,j-1)->h_j+1,i+1 e h_j,i+1:  h[:, 5]+h[:, 3] ?
            # uh(j+1)->h_j+2 e h_j+1:        h[:, 6]+h[:, 0] ?
            
            hbars = [[0,2],[2,1],[3,2],[2,4]] # ,[5,3],[6,0]?
            with torch.no_grad():
             self.linear.weight.fill_(0) 
             for i, n in enumerate(hbars):      
              self.linear.weight[i, n] = 0.5    # Atribui 0.5 nas colunas especificadas em n
                 
            self.bilinear = nn.Bilinear(linear_outdim, bilinear_indim, bilinear_outdim, bias=False)
            # bilinear para multiplicoçoes de uh(i),uh(i-1),uh(i+1) e vh(i),vh(i-1),vh(i+1)
            with torch.no_grad():
             self.bilinear.weight.fill_(0)
             for n in range(bilinear_outdim):
             # [índice_saída, índice_v1, índice_v2]
              self.bilinear.weight[n, n, n] = 1.0

            self.pre_gradient = nn.Bilinear(4, 3, 4, bias=False)
            # 1o tensor de entrada:new_z(,ji),new_z(i,j+1),topo(i,j),topo(i,j+1) # v:new_z(i,j),new_z(i+1,j),topo(i,j),topo(i+1,j)
            # 2o tensor de entrada:new_z(i,j),new_z(i,j+1),1                     # v:new_z(i,j),new_z(i+1,j),1 
            with torch.no_grad():
                self.pre_gradient.weight.fill_(0)
                self.pre_gradient.weight[0, 0, 0] = 1     # new_z(i)^2
                self.pre_gradient.weight[1, 1, 1] = 1     # new_z(i+1)^2
                self.pre_gradient.weight[2, 1, 2] = 1/Dx  # del_x(new_z,'h')
                self.pre_gradient.weight[2, 0, 2] = -1/Dx
                self.pre_gradient.weight[3, 2, 2] = .5    # bar_x(topo,'h')
                self.pre_gradient.weight[3, 3, 2] = .5  
                
            #Gradiente de pressao
            #grad_ubar = -cff*(self.del_x(new_z*new_z,"h")/2+self.bar_x(torch.from_numpy(self.sw.eta_b),"h")*(self.del_x(new_z,"h"))) # inverte para bar e del_y em V
            self.pressure_gradient = nn.linear(3, 1, bias=False)
            #tensor de entrada: new_z(i)^2, new_z(i+1)^2, del_x(new_z,'h')*bar_x(topo,'h')
            cff = ...
            with torch.no_grad():
             self.pressure_gradient.weight.fill_(0)
             self.pressure_gradient.weight[0, 0] = +cff/(2*Dx)
             self.pressure_gradient.weight[0, 1] = -cff/(2*Dx)
             self.pressure_gradient.weight[0, 2] = -cff

            #### ADV ### 
            #UFx = (self.bar_x(uh,"u")*self.bar_x(u,"u")) # inverte x e y em V
            #UFe = (self.bar_x(vh,"v")*self.bar_y(u,"u")) # inverte x e y em V
            # para -UFx(j)  : (-uh(i,j-1)-uh(i,j))/2*(-u(i,j-1)-u(i,j))/2     # emv: (-vh(i-1,j)-vh(i,j))/2*(-v(i-1,j)-v(i,j))/2
            # para UFx(j+1) : (uh(i,j)+uh(i,j+1))/2*(u(i,j)+u(i,j+1))/2      # em v: (vh(i-1,j+1)+vh(i,j+1))/2*(v(i-1,j+1)+v(i,j+1))/2
            # para UFe(i,j)   : (vh(i,j+1)+vh(i,j))/2*(u(i+1,j)+u(i,j))/2     #em v: (vh(i+1,j)+vh(i,j))/2*(u(i,j+1)+u(i,j))/2 
            # para -UFe(i-1,j): (-vh(i-1,j+1)-vh(i-1,j))/2*(-u(i,j)-u(i-1,j))/2 # v: (-vh(i,j)-vh(i-1,j))/2*(-u(i-1,j+1)-u(i-1,j))/2 
            
            # Tensor de entrada: 0:uh(i,j-1)),1:uh(i,j),2:uh(i,j+1),
            #                    3:u(i,j-1),4:u(i,j),5:u(i,j+1),6:u(i+1,j),7:u(i-1,j),
            #                    8:vh(i,j+1),9:vh(i,j),10:vh(i-1,j+1),11:vh(i-1,j)
            self.pre_adv = nn.linear(12, 8, bias=False)
            with torch.no_grad():
                self.pre_adv.weight.fill_(0)
                self.pre_adv.weight[0,0] = -1 # -uh(i,j-1))-uh(i,j)
                self.pre_adv.weight[0,1] = -1 
                self.pre_adv.weight[1,4] = -1  # -u(i,j)-u(i,j-1)
                self.pre_adv.weight[1,3] = -1   
                self.pre_adv.weight[2,1] = 1  # uh(i,j)+uh(i,j+1)
                self.pre_adv.weight[2,2] = 1  
                self.pre_adv.weight[3,4] = 1 # u(i,j)+u(i,j+1)
                self.pre_adv.weight[3,5] = 1
                self.pre_adv.weight[4,8] = 1 # vh(i,j+1)+vh(i,j)
                self.pre_adv.weight[4,9] = 1
                self.pre_adv.weight[5,6] = 1  # u(i+1,j)+u(i,j)
                self.pre_adv.weight[5,4] = 1
                self.pre_adv.weight[6,10]= -1  # -vh(i-1,j+1)-vh(i-1,j)
                self.pre_adv.weight[6,11] = -1
                self.pre_adv.weight[7,4]= -1 # -u(i,j)-u(i-1,j)
                self.pre_adv.weight[7,7] = -1
                
            # 1o Tensor de entrada: -uh(i,j-1))-uh(i,j), uh(i,j)+uh(i,j+1), vh(i,j+1)+vh(i,j), -vh(i-1,j+1)-vh(i-1,j)
            # 2o tensor de entrada: -u(i,j)-u(i,j-1)   , u(i,j)+u(i,j+1)  , u(i+1,j)+u(i,j)  , -u(i,j)-u(i-1,j)
            self.adv = nn.Bilinear(4, 4, 1, bias=False)
            with torch.no_grad():
                self.adv.weight.fill_(0)
                for n in range(bilinear_outdim):
                 # [índice_saída, índice_v1, índice_v2]
                 #ADV_u = -self.del_x(UFx,"h")-self.del_y(UFe,"q") # IGUAL em V
                 # Soma todos as entradas multiplicadas na diagonal UFx(i+1),-UFx(i),UFe(j),-UFe(j-1)
                 self.adv.weight[0, n, n] = -.5/Dx 
                
            #### COR ### 
            #UFx = h*self.sw.f*self.bar_y(v,"v") # inverte para bar_x em V
            #1o tensor de entrada: h(i,j),h(i,j+1) # para v: h(i,j),h(i+1,j)
            #2o tensor de entrada: v(i-1,j),v(i,j),v(i,j+1),v(i-1,j+1) # para v: u(j-1,i),v(i,j),u(i+1,j),u(j-1,i+1)
            # UFx(j) = h(j)(v(i)+v(i-1))/2
            # UFx(j+1) = h(j+1)(v(i,j+1)+v(i-1,j+1))/2
            self.pre_cor = nn.Bilinear(2, 3, 2, bias=False)
            with torch.no_grad():
                self.pre_cor.weight.fill_(0)
                self.pre_cor.weight[0, 0, 0] = f*.5     # h(i,j)*f*v(i-1,j)/2
                self.pre_cor.weight[0, 0, 1] = f*.5     # h(i,j)*f*v(i,j)/2
                self.pre_cor.weight[1, 1, 2] = f*.5     # h(j+1)*f*v(i,j+1)/2
                self.pre_cor.weight[1, 1, 3] = f*.5     # h(j+1)*f*v(i-1,j+1)/2
            #Cor_u = self.bar_x(UFx,"h")         # inverte para bar_y em V
            # Tesor de entrada: UFx(j),UFx(j+1)
            self.cor = nn.linear(2, 1, bias=False)
            with torch.no_grad():
                self.cor.weight.fill_(0)
                self.cor.weight[0, 0] = .5 
                self.cor.weight[0, 1] = .5
                
            #### SUM ###
            #utend = grad_ubar  + torch.from_numpy(Cor_u + ADV_u) # Coriolis é negativo em V
            self.linear = nn.bilinear(3, 1, 1, bias=False)
            # 1o tensor de entrada: pressure_gradient,Cor,ADV
            # 2o tensor de entrada: 1, Dummy (1 for u and -1 for v)
            with torch.no_grad():
                self.linear.weight.fill_(0)
                self.linear.weight[0, 0,0] = 1  
                self.linear.weight[0, 1,1] = 1 
                self.linear.weight[0, 2,0] = 1   
            
            self.bn0 = nn.BatchNorm1d(6) #, affine=False
            self.FC_input = nn.Linear(6, self.hidden_dim, bias=False)
            self.dropout = nn.Dropout(dropHidden)
            
        elif self.typeB:
            # Physics Informed Encoder for AB3 #
            linear_indim = 5 # vetores h
            linear_udim = 2 # vetores u e v
            linear_outdim = 4 # h_bar
            bilinear_indim = 4 # momentum
            bilinear_outdim = 4 # momentum
             # Temporal means (rhs) #
            self.beta = nn.Parameter(torch.tensor(0.28))
            # Linear and bilinear for spatial means, Uh/Vh and spatial derivatives #
            self.linear = nn.Linear(linear_indim, linear_outdim, bias=False) # linear para médias espaciais de H
            #            h[:, 0]+h[:, 2]         h[:, 2]+h[:, 1]
            #            h[:, 3]+h[:, 2]         h[:, 2]+h[:, 4]
            hbars = [[0,2],[2,1],[3,2],[2,4]]
            with torch.no_grad():
             self.linear.weight.fill_(0) 
             for i, n in enumerate(hbars):      
              self.linear.weight[i, n] = 0.5    # Atribui 0.5 nas colunas especificadas em n
  
            self.bilinear = nn.Bilinear(linear_outdim, bilinear_indim, bilinear_outdim, bias=False) # bilinear para multiplicoçoes de uh e vh
            if Dx is None:
                Dx = 1.0
            with torch.no_grad():
             self.bilinear.weight.fill_(0)
             for n in range(bilinear_outdim):
             # [índice_saída, índice_v1, índice_v2]
              self.bilinear.weight[n, n, n] = Dx
                
            self.bilinear2 = nn.Bilinear(bilinear_outdim+1,5,6, bias=False) # bilinear para dividir por dx,dy,dt
            #self.bilinear2 = nn.Bilinear(bilinear_outdim+1,5,1, bias=False) # bilinear
            print('Const',Const)
            if Const is None:
                Const = 1
            with torch.no_grad():
             self.bilinear2.weight.fill_(0)
             self.bilinear2.weight[0, 0,1]   = -Const# + (((h_p1 + h_0)/2)*u_0 )/ dx
             self.bilinear2.weight[0, 1,1]   = Const # - (((h_0 + h_m1)/2)*u_m1)/ dx
             self.bilinear2.weight[0, 2,3]   = -Const# + (((h_p1 + h_0)/2)*v_0 )/ dy
             self.bilinear2.weight[0, 3,3]   = Const # - (((h_0 + h_m1)/2)*v_m1)/ dy
             self.bilinear2.weight[1, -1, 0] = 1.0 # dx
             self.bilinear2.weight[2, -1, 1] = 1.0 # 1/dx
             self.bilinear2.weight[3, -1, 2] = 1.0 # dy
             self.bilinear2.weight[4, -1, 3] = 1.0 # 1/dy
             self.bilinear2.weight[5, -1, 4] = 1.0 # dt
            
            self.bn0 = nn.BatchNorm1d(6) #, affine=False
            self.FC_input = nn.Linear(6, self.hidden_dim, bias=False)
            self.dropout = nn.Dropout(dropHidden)
        
        elif self.typeC:
            self.bn0 = nn.BatchNorm1d(self.input_dim, affine=False)
            self.FC_input = nn.Linear(self.input_dim, self.hidden_dim, bias=False)
            self.dropout1 = nn.Dropout(dropHidden)
        else: # Encoder Antigo, type= 'D' #
            self.FC_input = nn.Linear(self.input_dim, self.hidden_dim, bias=False) # O input agora vem das novas camadas
            self.dropout1 = nn.Dropout(dropHidden)
            self.bilinear = nn.Bilinear(self.hidden_dim, self.hidden_dim, self.hidden_dim)
            self.dropoutbi = nn.Dropout(dropHidden)
        
        for j in range(self.ALPHAx):
            self.hiddenlayers.append( nn.Linear(self.hidden_dim, self.hidden_dim, bias=False) )
            self.dropouts.append( nn.Dropout(dropHidden) )

        # ENCODER EXIT
        self.FC_input2 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.dropout2 = nn.Dropout(dropHidden)
        self.FC_mean  = nn.Linear(self.hidden_dim, latent_dim, bias=False)
        self.FC_var   = nn.Linear (self.hidden_dim, latent_dim)
        
    def forward(self, x):
        
        # SINCE THE AE USES LINEAR LAYERS AND NO CONVS       #
        # IT CAN BE A 2D VECTOR WITH SHAPE [BATCH,INPUT_DIM] #
        x = x.view(-1, self.input_dim)
        x_ = None
        if self.typeA :
             # Reorganiza o vetor achatado em [Batch, 5 vizinhos, 4 variáveis por vizinho]
            x_h = x[:, 0:20].view(-1, 5, 4)
            x_u = x[:, 20:29].view(-1, 3, 3)
            x_v = x[:, 29:38].view(-1, 3, 3)
            dx,dy,dt = x[:, -3],x[:, -2],x[:, -1]
            
            # Cria médias temporais rhs (aplicando parametro beta)
            h2 = x_h[:, :, 0]  # Todas as h2 de todos os vizinhos (ij, j-1, j+1, i-1, i+1)
            h1 = x_h[:, :, 1]  # Todas as h1
            h0 = x_h[:, :, 2]  # Todas as h0
            topo = x_h[:, :, 3] # Todos os topos
            h = h2 * (1.5 + self.beta) - (0.5 + 2 * self.beta) * h1 + self.beta * h0 + topo
            u2 = x_u[:, :, 0]  # Todas as u2 de todos os vizinhos (ij, j-1,j+1)
            u1 = x_u[:, :, 1]  # Todas as u1
            u0 = x_u[:, :, 2]  # Todas as u0
            u = u2 * (1.5 + self.beta) - (0.5 + 2 * self.beta) * u1 + self.beta * u0
            v2 = x_v[:, :, 0]  # Todas as v2 de todos os vizinhos (ij i-1,i+1)
            v1 = x_v[:, :, 1]  # Todas as v1
            v0 = x_v[:, :, 2]  # Todas as v0
            v = v2 * (1.5 + self.beta) - (0.5 + 2 * self.beta) * v1 + self.beta * v0
                        
            uv   = torch.concat((u,v),dim=1)
            res  = torch.concat((dx.view(-1, 1),1/dx.view(-1, 1),dy.view(-1, 1),
                                 1/dy.view(-1, 1),dt.view(-1, 1)),dim=1)

            hbar = self.linear(h)
  
            h_ = self.bilinear(hbar,uv) # uh e vh
            
        elif self.typeB:
            # Physics Informed Encoder. #
            
            # Reorganiza o vetor achatado em [Batch, 5 vizinhos, 4 variáveis por vizinho]
            x_h = x[:, 0:20].view(-1, 5, 4)
            x_u = x[:, 20:26].view(-1, 2, 3)
            x_v = x[:, 26:32].view(-1, 2, 3)
            dx,dy,dt = x[:, -3],x[:, -2],x[:, -1]
            
            # Cria médias temporais rhs (aplicando parametro beta)
            h2 = x_h[:, :, 0]  # Todas as h2 de todos os vizinhos (ij, j-1, j+1, i-1, i+1)
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
                        
            uv   = torch.concat((u,v),dim=1)
            res  = torch.concat((dx.view(-1, 1),1/dx.view(-1, 1),dy.view(-1, 1),
                                 1/dy.view(-1, 1),dt.view(-1, 1)),dim=1)

            hbar = self.linear(h)
  
            h_ = self.bilinear(hbar,uv) # uh e vh
            h_ = torch.concat((h_,torch.ones(hbar.shape[0], 1)),dim=1)

            h_ = self.bilinear2(h_,res)
            ############### # constant network resolution for testing # ###############
            #batch_size = x.shape[0]                                                   #
            #dx     = torch.full((batch_size, 1), 200157.7794640436, device=x.device)  #
            #inv_dx = 1 / dx                                                           #
            #dt     = torch.full((batch_size, 1), 360.0, device=x.device)              #
            #res_tensor = torch.cat([dx, inv_dx, dx, inv_dx, dt], dim=1)               #
            ##print(res_tensor.shape)                                                  #
            #h_[:,1:] = res_tensor                                                     #
            ##############################(comment during training )###################
            
            h_ = self.bn0(h_)

            h_ = self.FC_input(h_)
            h_ = self.activation(h_)
            h_ = self.dropout(h_)
            
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
            # FORMULA DIRETA:
            #h_p1 = h[:, 0]
            #h_m1 = h[:, 1]
            #h_0  = h[:, 2]
            #u_0  = u[:, 0]
            #u_m1 = u[:, 1]
            #term_x = ( ((h_p1 + h_0)/2)*u_0 - ((h_0 + h_m1)/2)*u_m1 ) / dx
            #            h[:, 0]+h[:, 2]         h[:, 2]+h[:, 1]
            #h_p1 = h[:,3]
            #h_m1 = h[:,4]
            ## Pegamos os vizinhos de U
            #v_0  = v[:, 0]
            #v_m1 = v[:, 1]
            #vh = ((h_p1 + h_0)/2)*v_0
            ## Cálculo em passo único (exatamente como o seu Pre_terms3a)
            #term_y = ( ((h_p1 + h_0)/2)*v_0 - ((h_0 + h_m1)/2)*v_m1 ) / dy
            #            h[:, 3]+h[:, 2]         h[:, 2]+h[:, 4]
            #htend = -(term_x + term_y)
            
        elif self.typeC:
            # Normalizes Input inside network #
            ############### # constant network resolution for testing # ###############
            #batch_size = x.shape[0]                                                   #
            #dx     = torch.full((batch_size, 1), 200157.7794640436, device=x.device)  #
            #inv_dx = 1 / dx                                                           #
            #dt     = torch.full((batch_size, 1), 360.0, device=x.device)              #
            #res_tensor = torch.cat([dx, inv_dx, dx, inv_dx, dt], dim=1)               #
            #print(res_tensor.shape)                                                   #
            #x[:,1:] = res_tensor                                                      #
            ##############################(comment during training )###################
            
            x = self.bn0(x)

        # Encoder Antigo das redes S26WE300Jet... #
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
                 dropKoop=0.10,dropHidden=0.35,type='D',Const=None):
        
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
        self.n_variables    = n_variables
        self.dynamics     = nn.ModuleList()
        self.backdynamics = nn.ModuleList()
        
        if not multidynamics:
            self.dynamics.append(dynamics(b,bx, init_scale,dropKoop=dropKoop))
            self.backdynamics.append(dynamics_back(b,bx, self.dynamics[0],dropKoop=dropKoop))
        if not multiencoder:
            self.encoder.append(VencoderNet(dim, n, Bound_dim, b, hidden_dim = alpha, ALPHAx = alphax,
                                            ActFunc=ActFunc,dropHidden=dropHidden,type=type,Const=Const))
        
        for i in range(n_variables): # CADA VARIAVEL TEM UMA DINAMICA E DECODER NO KOOPMAN SPACE
            if multidynamics:
                self.dynamics.append(dynamics(b,bx, init_scale,dropKoop=dropKoop))
                self.backdynamics.append(dynamics_back(b,bx, self.dynamics[i],dropKoop=dropKoop))
            if multiencoder:
                self.encoder.append(VencoderNet(dim, n, Bound_dim, b, hidden_dim = alpha,ALPHAx = alphax,
                                                ActFunc=ActFunc,dropHidden=dropHidden,type=type,Const=Const))
            self.decoder.append(VdecoderNet(dim_out[i], b, hidden_dim = beta, BETAx = betax, 
                                            ActFunc = ActFunc,dropHidden=dropHidden))

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
    def __init__(self, layers, dim, n, Bound_dim, ActFunc, dropHidden):

        # Rede multilayer - com função de ativação.

        super(MultilayerNetwork, self).__init__()
        
        self.input_dim = (dim + Bound_dim) * n
        print('dimensao do vetor de entrada na rede = ', self.input_dim)
        print('Dropout = ', dropHidden)
        
        self.activation = ActFunc
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
    def __init__(self, layers, dim, dim_out, n, Bound_dim, n_variables=1, linear = False, ActFunc = nn.Tanh(), dropHidden=0.40):

        super(ML, self).__init__()
        self.n_variables = n_variables
        ## Multi Layer Network ##
        if not linear:
            self.network = MultilayerNetwork(layers, dim, n, Bound_dim, ActFunc, dropHidden)
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
