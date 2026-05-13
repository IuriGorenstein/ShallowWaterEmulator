import torch
import torchvision
from torch import nn
from torch import autograd
from torch import optim
from torchvision import transforms, datasets
from torch.autograd import grad
from torch.utils.data import DataLoader, Dataset
from timeit import default_timer as timer

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import cartopy.crs as ccrs    # projections
import cartopy.feature as cf  # features
import xarray as xr
import numpy as np
import numpy.ma as ma
import importlib
import dask
import sys
from simple_colors import *

import Variational_Model
importlib.reload(Variational_Model)
from Variational_Model import *

import torch.nn.init as init

def sci_formatter(x, pos):
    return f"{x:.1e}"
 # Escala da normalização das funções Pega_Bits
escala = 1

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

def ChamaModelo(args,lstsq=False,type=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if lstsq:
        print('DMDc - Least Squares Model')
        model = DMD(n_variables=1)
        return model.to(device),'DMDc'
    else:
        if type == None:
            try:
                print('Tentando usar rede Variational KAE')
                model = VKAE(args.dim, args.dim_out, args.n, args.Bdim, args.bottleneck,
                             args.n_variables, args.alpha,args.beta, args.alphax, args.betax, 
                             args.bottleneckx, args.init_scale)
                return model.to(device),'KAE'
            except: # modelo multi layer
                try:
                    try:
                        print('tentando usar REDE MULTILAYER')
                        model = ML(args.layers, args.dim, args.dim_out, args.n,
                                   args.Bdim, args.n_variables,linear=args.linear)
                        #Antigo VAriational Model ML...
                        #model = ML(args.layers,args.dim, args.dim_out, args.n, args.Bdim, args.bottleneck,
                        #        args.n_variables, args.steps, args.steps_back, args.alpha,
                        #        args.beta, args.alphax, args.betax, args.bottleneckx, args.init_scale,args.linear)
    
                        return model.to(device),'ML'
                    except:
                    # sem layers salvas...
                        layers = []
                        for _ in range(args.alphax):
                            layers.append(args.alpha)
                        model = ML(layers,args.dim, args.dim_out, args.n, args.Bdim, args.bottleneck,
                                args.n_variables, args.steps, args.steps_back, args.alpha,
                                args.beta, args.alphax, args.betax, args.bottleneckx, args.init_scale,linear=True)
                        return model.to(device),'ML sem layers'
                        
                        
                except: # modelo KAE, mas salvo sem alphax,betax,bootleneckx
                    print('REDE VKAE sem parametros salvos')
                    model = VKAE(args.dim, args.dim_out, args.n, args.Bdim, args.bottleneck,
                                 args.n_variables, args.alpha,args.beta, args.alphax, args.betax, 
                                 args.bottleneckx, args.init_scale)
                    
                    return model.to(device),'KAE antiga'
        else:
            if args.NN=='Beta':
                print('Beta Net')
                model = BetaNet(args.dim, args.n, args.Bdim)
                return model.to(device),'BetaNet'
            
            print('Lie AE')
            model = VLieAE(args.dim, args.dim_out, args.n, args.Bdim, args.bottleneck,
                         args.n_variables, args.alpha,args.beta, args.alphax, args.betax, 
                         args.bottleneckx, args.init_scale,multiencoder=False,
                         multidynamics = False, ActFunc=nn.ReLU6(),dropKoop=0.2,dropHidden=0.4,
                         type=type)
            return model.to(device),'LAE'
            
    
def CHAMA(PATH,lstsq=False,type=None):
    # PARA A Multi Variable KAE
    #args, state, Epocas = torch.load(PATH,map_location='cpu')
    # PARA A SINGLE VARIABLE KAE
    
    args, modelo, Epocas, Epocas_treino = torch.load(PATH,map_location='cpu')
    # args e modelo podem ser listas com os modelos e argumentos respectivos.
    if 1:
        state = modelo.state_dict()
        KAE,type = ChamaModelo(args,lstsq,type)
        if not lstsq:
            KAE.load_state_dict(state)
            KAE.eval()
            if type =='ML':
                print('\n ############### Rede ML: layers-',args.layers,' ################### \n')
            else:
                print('\n ############### Rede KAE: a-' +str(args.alpha) + ', b-' +str(args.beta)+ ', K-'+str(args.bottleneck)+' ################### \n')
    
            #==============================================================================
            # Model summary
            #==============================================================================
            print('**** Setup SSH ****')
            print('Total params: %.2fM' % (sum(p.numel() for p in KAE.parameters())/1000000.0))
            print('Total params: %.2fk' % (sum(p.numel() for p in KAE.parameters())/1000.0))
            print('************')   
        else:
            KAE.dynamics=copy.deepcopy(args)
            KAE = KAE.to(device)
            print('\n ############### Rede DMDc - least Squares Algorithm ################### \n')
        
        return KAE, args, modelo, Epocas, Epocas_treino
    
def set_seed(seed=0):
    """Set one seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_device():
    """Get a gpu if available."""
    if torch.cuda.device_count()>0:
        device = torch.device('cuda')
        print("Connected to a GPU")
    else:
        print("Using the CPU")
        device = torch.device('cpu')
    return device


def add_channels(X):
    if len(X.shape) == 2:
        return X.reshape(X.shape[0], 1, X.shape[1],1)

    elif len(X.shape) == 3:
        return X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])

    else:
        return "dimenional error"


def weights_init(m):
    if isinstance(m, nn.Linear):
        if m.weight is not None:
            init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0.0)
            
def Cinetic_Energy(U,V):
    # Uses U and V currents
    # returns the total Cinetic Energy and the Mean
    
    return np.nansum(np.sqrt(np.square(U)+np.square(V))/2), np.nanmean(np.sqrt(np.square(U)+np.square(V))/2)

def Volume(SSH,dx,dy):
    # Uses the Sea surface height and spatial resolution
    # returns the total mass above sea level (if density = 1)
    
    return np.nansum(SSH*dx*dy)

def Potential_Energy(SSH,dx,dy):
    # Uses the Sea surface height and spatial resolution
    # returns the total Potential Energy and the Mean
    
    g = 9.81 #m/s²
    
    return np.nansum(g*np.square(SSH)*dx*dy/2), np.nanmean(g*np.square(SSH)*dx*dy/2)

#################################################################################################
 ################################ SETUP PARA RESET TRAINNG ####################################
#################################################################################################

def progressbar(it, prefix="", size=40, file=sys.stdout):
    count = len(it)
    def show(j):
        x = int(size*j/count)
        space=int((size-18)/2)
        a='#'*space
        mensagem = a + ' VAI CORINTHIANS! ' + a
        file.write("%s[%s%s] %i/%i\r" % (prefix, mensagem[0:x], "."*(size-x), j, count)) #"#"*x
        file.flush()        
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    file.write("\n")
    file.flush()


def progressbarZip(x,y,count,prefix="", size=40, file=sys.stdout):
    c=0
    def show(j):
        x = int(size*j/count)
        space=int((size-18)/2)
        a='#'*space
        mensagem = a + ' VAI CORINTHIANS! ' + a
        file.write("%s[%s%s] %i/%i\r" % (prefix, mensagem[0:x], "."*(size-x), j, count)) #"#"*x
        file.flush()  
    show(c)
    for a, b in zip(x,y):
        c+=1
        yield a,b
        show(c)
    file.write("\n")
    file.flush()

def vetoriza(lista_de_vetores):
    nova_lista = []
    
    for vetor in lista_de_vetores:
        for l in range(vetor.shape[2]):
            nova_lista.append( vetor[:,:,l].values.reshape(( vetor.shape[0],1 )) )
    return nova_lista

def Vetor_Mean_Std(Input_vector,imprime=False):
    
    counter = -1
    Mean_Std = np.zeros((2,len(Input_vector))) # variaveis + Topografia, Coriolis, dx,dy, dt
    if imprime:
        fig = plt.figure(figsize=(2*len(Input_vector),3))
    for var in Input_vector:
        counter +=1
        Mean_Std[0,counter] =  np.nanmean(var) # 0 # 
        Mean_Std[1,counter] =  np.nanstd(var)  # 1 # 
        
        if imprime:
            ax = fig.add_subplot(1,len(Input_vector),counter+1)
            print('VAR',counter,' - max: ',np.max(var),',min: ',np.min(var), ',media: ', np.nanmean(var), ',std: ',np.nanstd(var) )
            ax.hist(var) 
        
    return Mean_Std    

def Normaliza_Tensor(Input_tensor,Mean_Std,imprime=False,inverse=False):
    
    if imprime:
        fig = plt.figure(figsize=(2*len(Input_vector),3))

    T=1
    if inverse:
        T=-1
    for counter in range(Input_tensor.shape[2]):
        Input_tensor[:,0,counter,0] = Input_tensor[:,0,counter,0]-(Mean_Std[0,counter]*T)
        if Mean_Std[1,counter]!=0:
            if inverse:
                Input_tensor[:,0,counter,0] = Input_tensor[:,0,counter,0] * Mean_Std[1,counter]
            else:     
                Input_tensor[:,0,counter,0] = Input_tensor[:,0,counter,0] / Mean_Std[1,counter]
        
        if imprime:
            ax = fig.add_subplot(1,Input_tensor.shape[2],counter+1)
            print('VAR',counter,' - max: ',np.max(Input_tensor[counter,0,counter,0]),',min: ',
                  np.min(Input_tensor[:,0,counter,0]), ',media: ', np.nanmean(Input_tensor[:,0,counter,0]),
                  ',std: ',np.nanstd(Input_tensor[:,0,counter,0])
                 )
            ax.hist(Input_tensor[:,0,counter,0]) 
        
    return Input_tensor  

def Normaliza_Vetor(Input_vector,Mean_Std,imprime=False,inverse=False):
    
    if imprime:
        fig = plt.figure(figsize=(2*len(Input_vector),3))
    T=1
    if inverse:
        T=-1
        
    for counter in range(len(Input_vector)):
        Input_vector[counter] = Input_vector[counter]-(Mean_Std[0,counter]*T)
        if Mean_Std[1,counter]!=0:
            if inverse:
                Input_vector[counter] = Input_vector[counter] * Mean_Std[1,counter]
            else:
                Input_vector[counter] = Input_vector[counter] / Mean_Std[1,counter]
        
        if imprime:
            ax = fig.add_subplot(1,len(Input_vector),counter+1)
            print('VAR',counter,' - max: ',np.max(Input_vector[counter]),',min: ',
                  np.min(Input_vector[counter]), ',media: ', np.nanmean(Input_vector[counter]), ',std: ',np.nanstd(Input_vector[counter]) )
            ax.hist(Input_vector[counter]) 
        
    return Input_vector   

def Pega_Mean_Std(Simulador_variables,imprime=False):
    #########################################################################
    # Tira Média e desvio padrão de todos os domínios em Simulador_Variables#
    #########################################################################
    print('Calculando Médias e desvios padrões do Sistema')
    
    variaveis_sys = [subsim[0] for subsim in Simulador_variables[:]]
    Mean_Std = np.zeros((2,len(variaveis_sys[0])+5)) # variaveis + Topografia, Coriolis, dx,dy, dt
    
    #####################################
    #####  Variaveis Zeta,Ubar e Vbar ###
    #######################
    for var in range( len(variaveis_sys[0]) ):  # sao 3 variaveis (zeta, u, v)
        for n,subsys in enumerate(variaveis_sys):  # sao 9 dominios diferentes
            if n > 0:
                data = np.concatenate((data,subsys[var].reshape(-1)))
            else:
                data = subsys[var].reshape(-1)
        if imprime:
            fig = plt.figure(figsize=(10,2))
            ax = fig.add_subplot(1,len(variaveis_sys[0]),var+1)
            print('VAR',var,' - max: ',np.max(data),',min: ',np.min(data), ',media: ', np.nanmean(data), ',std: ',np.nanstd(data) )
            ax.hist(data) 

        
        Mean_Std[0,var] = 0 #  np.nanmean(data) #
        Mean_Std[1,var] = 1 #  np.nanstd(data)  #
        
    var+=1
    ##############
    # Topografia #
    ##############
    Topo = [subsim[3] for subsim in Simulador_variables]
    for n,systemas in enumerate(Topo): # sao 9 dominios diferentes
        if n > 0:
            data = np.concatenate((data,systemas.reshape(-1)))
        else:
            data = systemas.reshape(-1)
       
    if imprime:
        fig = plt.figure(figsize=(10,2))
        ax = fig.add_subplot(1,2,1)
        print('TOPOGRAFIA - max: ',np.max(data),',min: ',np.min(data), ',media: ', np.nanmean(data), ',std: ',np.nanstd(data) )
        p = ax.hist(data) 
    Mean_Std[0,var] = 0 # np.nanmean(data)
    Mean_Std[1,var] = 1 # np.nanstd(data) 
    
    var+=1
    
    ############
    # Coriolis #
    ############
    Cori = [subsim[4] for subsim in Simulador_variables]
    for n,systemas in enumerate(Cori): # sao 9 dominios diferentes
        if n > 0:
            data = np.concatenate((data,systemas.reshape(-1)))
        else:
            data = systemas.reshape(-1)
       
    if imprime:
        ax = fig.add_subplot(1,2,2)
        print('CORIOLIS - max: ',np.max(data),',min: ',np.min(data), ',media: ', np.nanmean(data), ',std: ',np.nanstd(data) )
        p = ax.hist(data)
    Mean_Std[0,var] = 0 # np.nanmean(data)
    Mean_Std[1,var] = 1 # np.nanstd(data) 
    
    var+=1
    ######
    # dx #
    ######
    DX = [subsim[16] for subsim in Simulador_variables]
    for n,systemas in enumerate(DX): # sao 9 dominios diferentes
        if n > 0:
            data = np.concatenate((data,systemas.reshape(-1)))
        else:
            data = systemas.reshape(-1)

    if imprime:
        fig = plt.figure(figsize=(10,2))
        ax = fig.add_subplot(1,2,1)
        print('DX - max: ',np.max(data),',min: ',np.min(data), ',media: ', np.nanmean(data), ',std: ',np.nanstd(data) )
        p = ax.hist(data)
    Mean_Std[0,var] = np.nanmean(data)
    Mean_Std[1,var] = np.nanstd(data) 

    var+=1
    ######
    # dy #
    ######
    DY = [subsim[17] for subsim in Simulador_variables]
    for n,systemas in enumerate(DY): # sao 9 dominios diferentes
        if n > 0:
            data = np.concatenate((data,systemas.reshape(-1)))
        else:
            data = systemas.reshape(-1)

    if imprime:
        fig = plt.figure(figsize=(10,2))
        ax = fig.add_subplot(1,2,1)
        print('DY - max: ',np.max(data),',min: ',np.min(data), ',media: ', np.nanmean(data), ',std: ',np.nanstd(data) )
        p = ax.hist(data)
        
    Mean_Std[0,var] = np.nanmean(data)
    Mean_Std[1,var] = np.nanstd(data) 
    
    var+=1
    
    ######
    # dt #
    ######
    data= []
    DT = [subsim[18] for subsim in Simulador_variables]
    for n,systemas in enumerate(DT): # sao 9 dominios diferentes
        data.append(systemas)
    data = np.array(data)
            
    if imprime:
        fig = plt.figure(figsize=(10,2))
        ax = fig.add_subplot(1,2,1)
        print('DT - max: ',np.max(data),',min: ',np.min(data), ',media: ', np.nanmean(data), ',std: ',np.nanstd(data) )
        p = ax.hist(data)
    Mean_Std[0,var] = np.nanmean(data)
    Mean_Std[1,var] = np.nanstd(data) 
    print('Shape do vetor Mean_Std' , Mean_Std.shape)

    return Mean_Std
    

def Pega_Mean_Std_Bulk(Simulador_variables,imprime=False):
    #########################################################################
    # Tira Média e desvio padrão de todos os domínios em Simulador_Variables#
    #########################################################################
    print('Calculando Médias e desvios padrões do Bulk')
    
    variaveis_sys = [subsim[2] for subsim in Simulador_variables[:]]
    Mean_Std = np.zeros((2,len(variaveis_sys[0]))) # variaveis, costumam ser 9...
    
    for var in range( len(variaveis_sys[0]) ):  # sao 3 variaveis (zeta, u, v)
        for n,subsys in enumerate(variaveis_sys):  # sao 9 dominios diferentes
            if n > 0:
                data = np.concatenate((data,subsys[var].reshape(-1)))
            else:
                data = subsys[var].reshape(-1)
        if imprime:
            if var == 0:
                fig = plt.figure(figsize=(10,5))
            ax = fig.add_subplot(3,4,var+1)
            print('VAR',var,' - max: ',np.max(data),',min: ',np.min(data), ',media: ', np.nanmean(data), ',std: ',np.nanstd(data) )
            ax.hist(data) 
        Mean_Std[0,var] = np.nanmean(data) # 0
        Mean_Std[1,var] = np.nanstd(data) # 1

    return Mean_Std

def Roda_90(sistemas):
    k=1 # gira 90 graus no sentido do relogio
    
    # inverte 0:new_system, 1:bot, 2:new_top, 3:Topografia, 4:Coriolis 5:mascara, 10:lon
    new_system = copy.deepcopy(sistemas[0]) # [var,time,lat,lon)
    old_system = copy.deepcopy(sistemas[0]) # [var,time,lat,lon)
    new_system[1] = old_system[2] # ubar vira vbar
    new_system[2] = -old_system[1] # vbar vira -ubar

    #new_system    = new_system.swapaxes(2, 3)
    new_system   = np.rot90(new_system, k=k,  axes=(2, 3))
    
    if len(sistemas[1])>0:
        bot  = copy.deepcopy(sistemas[1]) # [var,time,lat,lon)
        bot[1] = sistemas[1][2,:] # ubar vira vbar
        bot[2] = -sistemas[1][1] # vbar vira -ubar
        #bot    = bot.swapaxes(2, 3)
        bot = np.rot90(bot, k=k,  axes=(2, 3))      
    else:
        bot =[]
        
    if len(sistemas[2])>0: 
        new_top = copy.deepcopy(sistemas[2]) # [var,time,lat,lon)
        #new_top = new_top.swapaxes(2, 3)
        new_top = np.rot90(new_top, k=k,  axes=(2, 3))    
    else:
        new_top =[]
        
    Topografia = copy.deepcopy(sistemas[3]) # [lat,lon)
    #Topografia = Topografia.T
    Topografia = np.rot90(Topografia, k=k)

    if sistemas[20] == 0: # Coriolis esta na horizontal
        #primeiro giro
        Coriolis = -copy.deepcopy(sistemas[4]) # é preciso inverter orientação da força de Coriolis...
        #Coriolis = Coriolis.T
        Coriolis = np.rot90(Coriolis, k=k)
        CoriolisDummy = 1 # Girando o sistema, Coriolis agora esta na vertical.
    else: # Coriolis esta na vertical
        Coriolis = copy.deepcopy(sistemas[4])
        #Coriolis = Coriolis.T
        Coriolis = np.rot90(Coriolis, k=k)
        CoriolisDummy = 0 # Girando o sistema o Coriolis vai para a horizontal

    mascara = copy.deepcopy(sistemas[5]) # [lat,lon)
    #mascara = mascara.T
    mascara = np.rot90(mascara, k=k)

    lat =-copy.deepcopy(sistemas[10]) # INVERTE lat e lon
    #lat = lat.T
    lat = np.rot90(lat, k=k)
    #lat = lat[::-1,:]
    lon = copy.deepcopy(sistemas[9]) #
    #lon = lon.T
    lon = np.rot90(lon, k=k)
    #lon = lon
    #lon = lon[:,::-1]
    dx = copy.deepcopy(sistemas[17]) # Inverte dx com dy
    dy = copy.deepcopy(sistemas[16]) #
    dx = np.rot90(dx, k=k)
    dy = np.rot90(dy, k=k)
    #dx = dx.T
    #dy = dy.T

    return [new_system, bot, new_top[:,:,:,:], Topografia[:,:],
            Coriolis[:,:], mascara[:,:],sistemas[6], sistemas[7],
            sistemas[8], lat[:,:],lon[:,:],sistemas[11],
            sistemas[12],sistemas[13],sistemas[14],sistemas[15],
            dx,dy,sistemas[18],sistemas[19],CoriolisDummy,sistemas[21]
           ]


def Flip_Simulador_variables(sistemas,mode):
    ## HA AQUI UMA QUESTAO. Além de INVERTER AS MATRIZES é preciso INVERTER LAT/LON de forma consistente. ##
    ### Para checar se as velocidades e vorticidades estão corretas é preciso protar o sistema em coordenadas de mapa (lat/lon) ###
    ### E  coordenadas do tensor do sistema (x,y) onde o simulador realiza os cortes do input vector ###
    
    # Retorna valores flipados em 180 graus apartir das definicoes das variaveis em PegaSistema3().
    # sistemas é um dos dominios do Simulador_variables (que é uma lista de domínios).
    # note que para rotacionar o sistema em 90 graus preciso de uma matriz representando a força de coriolis no eixo vertical
    # desse modo, o dominio poderia ser rotacionado e força de coriolis poderia ser representada nesse eixo.

    print('Coriolis Dummy',sistemas[20])
    if mode == 'horizontal':
        # inverte 0:new_system, 1:bot, 2:new_top, 3:Topografia, 4:Coriolis 5:mascara, 10:lon
            new_system = copy.deepcopy(sistemas[0])
            #new_system = copy.deepcopy(sistemas[0][:,:,:,::-1]) # [var,time,lat,lon) ANTIGO
            for i in [0,1,2]: # inverto lon
                new_system[i] = new_system[i][:,:,::-1]
                
            new_system[1] = - new_system[1]
            #new_system[1] = - new_system[1] # inverti o lon e estou invertendo as velocidades ubar ANTIGO
        
            if len(sistemas[1])>0:
                bot = sistemas[1] # [var,time,lat,lon)
                for B in bot:
                    B = B[:,:,::-1]
            else:
                bot =[]
            if len(sistemas[2])>0: 
                new_top = sistemas[2]
                for T in new_top:
                    T=T[:,:,::-1] # [var,time,lat,lon)
            else:
                new_top =[]
            Topografia = copy.deepcopy(sistemas[3][:,::-1]) # [lat,lon)
            Coriolis   = copy.deepcopy(sistemas[4][:,::-1]) # é preciso inverter a direção de Coriolis pois inverti lon
            if sistemas[20] == 0: # Coriolis na posição horizontal
                Coriolis = - Coriolis
            mascara = copy.deepcopy(sistemas[5][:,::-1]) # [lat,lon)
            lat = copy.deepcopy(sistemas[9][:,::-1]) # [lat,lon) era: [:,::-1]
            lon =-copy.deepcopy(sistemas[10][:,::-1]) # INVERTO LON era [:,::-1]
            dx  = copy.deepcopy(sistemas[16][:,::-1]) # [lat,lon)
            dy  = copy.deepcopy(sistemas[17][:,::-1]) # [lat,lon)

    elif mode == 'vertical':
            new_system = copy.deepcopy(sistemas[0]) # [var,time,lat,lon)
            for i in [0,1,2]: # inverto lat
                new_system[i] = new_system[i][:,::-1,:]

            new_system[2] = - new_system[2] # inverti o lat e estou invertendo as velocidades vbar

            if len(sistemas[1])>0:
                bot = sistemas[1] # [var,time,lat,lon)
                for B in bot:
                    B = B[:,::-1,:]
            else:
                bot =[]
            if len(sistemas[2])>0: 
                new_top = sistemas[2]
                for T in new_top:
                    T=T[:,::-1,:] # [var,time,lat,lon)
            else:
                new_top =[]
            Topografia = copy.deepcopy(sistemas[3][::-1,:]) # [lat,lon)
            Coriolis   = copy.deepcopy(sistemas[4][::-1,:]) # [lat,lon)
            if sistemas[20] == 0: # MUITO CUIDADO. QUANDO INVERTER OU NAO O Coriolis?
                Coriolis = - Coriolis
            mascara = copy.deepcopy(sistemas[5][::-1,:]) # [lat,lon)
            lat =-copy.deepcopy(sistemas[9][::-1,:]) # INVERTO A LAT
            lon = copy.deepcopy(sistemas[10][::-1,:]) # [lat,lon) era [::-1,:]
            dx  = copy.deepcopy(sistemas[16][::-1,:]) # [lat,lon)
            dy  = copy.deepcopy(sistemas[17][::-1,:]) # [lat,lon)
        
    return [new_system, bot, new_top, Topografia[:,:],
            Coriolis[:,:], mascara[:,:],sistemas[6], sistemas[7],
            sistemas[8], lat[:,:],lon[:,:],sistemas[11],
            sistemas[12],sistemas[13],sistemas[14],sistemas[15],
            dx,dy,sistemas[18],sistemas[19],sistemas[20],sistemas[21]
            ]


def transform_field_2d(field, rotate90=False, flip_vertical=False, flip_horizontal=False):
    """
    Applies spatial transformation to a 2D field.
    
    Parameters:
      field: 2D numpy array.
      rotate90: If True, rotate by 90° counterclockwise.
      flip_vertical: If True, flip vertically (mirror top-bottom).
      flip_horizontal: If True, flip horizontally (mirror left-right).
    
    Returns:
      Transformed 2D numpy array.
    """
    new_field = field.copy()
    if rotate90:
        new_field = np.rot90(new_field, k=1)  # rotates counterclockwise 90 degrees
    if flip_vertical:
        new_field = np.flip(new_field, axis=0)
    if flip_horizontal:
        new_field = np.flip(new_field, axis=1)
    return new_field

def transform_field_4d(field, rotate90=False, flip_vertical=False, flip_horizontal=False):
    """
    Applies spatial transformation to a 4D field.
    Assumes the field has shape [time, variables, lat, lon].
    
    Parameters:
      field: 4D numpy array.
      rotate90: If True, rotate spatial dimensions by 90° counterclockwise.
      flip_vertical: If True, flip vertically (lat axis).
      flip_horizontal: If True, flip horizontally (lon axis).
    
    Returns:
      Transformed 4D numpy array.
    """
    new_field = field.copy()
    if rotate90:
        new_field = np.rot90(new_field, k=1, axes=(-2, -1))
    if flip_vertical:
        new_field = np.flip(new_field, axis=2)
    if flip_horizontal:
        new_field = np.flip(new_field, axis=3)
    return new_field

def augment_data(sistemas,
                 rotate90=False, flip_vertical=False, flip_horizontal=False,
                 transpose_coords=False):
    
    if (transpose_coords and flip_vertical)+transpose_coords and flip_horizontal:
        print('Error, transpose_coords should not be done with a flip, only a rotation')
        return 
    """
    Augments the input dataset and associated 2D fields by rotating (90° counterclockwise)
    and/or flipping (vertical and/or horizontal).
    
    For the dataset (shape [time, variables, lat, lon]):
      - The scalar fields (e.g. free surface) are simply spatially re-ordered.
      - The velocity components (assumed at indices 1 for u and 2 for v)
        are re-ordered and then adjusted according to the transformation.
    
    For the Coriolis field (f_field), because it is a pseudo‑scalar, it is multiplied by 
    a parity factor of (-1) if an odd number of flips is applied.
    
    For the grid spacing arrays (dx, dy), if a rotation is applied the roles are swapped.
    
    Parameters:
      dataset: 4D numpy array with shape [time, variables, lat, lon].
      lat, lon, topo, f_field, dx, dy: 2D numpy arrays (shape [lat, lon]) for coordinates,
         topography, Coriolis parameter, and grid spacings.
      rotate90: If True, apply a 90° counterclockwise rotation.
      flip_vertical: If True, apply a vertical (north–south) flip.
      flip_horizontal: If True, apply a horizontal (east–west) flip.
    
    Returns:
      A tuple with the augmented (dataset, lat, lon, topo, f_field, dx, dy).
    """

    # -------------------------------
    # 1. Spatially re-order all fields
    # -------------------------------
    new_dataset = sistemas[0].copy()
    new_lat     = sistemas[9].copy()
    new_lon     = sistemas[10].copy()
    new_topo    = sistemas[3].copy()
    new_mascara = sistemas[5].copy()
    new_f_field = sistemas[4].copy()
    new_dx      = sistemas[16].copy()
    new_dy      = sistemas[17].copy()
    
    if transpose_coords:
        # --- Pure Transposition Mode ---
        # For the 4D dataset, swap the spatial dimensions:
        new_dataset = np.transpose(new_dataset, (0, 1, 3, 2))
        # For the 2D fields, simply transpose:
        new_lat     = new_lat    .T
        new_lon     = new_lon    .T
        new_topo    = new_topo   .T
        new_mascara = new_mascara.T
        new_f_field = new_f_field.T
        new_dx      = new_dx     .T
        new_dy      = new_dy     .T
        if rotate90:
            # After a 90° rotation, the roles of dx and dy swap.
            new_dx, new_dy = new_dy, new_dx
        
    else:
        # -------------------------------
        # 1. Spatially re-order all fields
        # -------------------------------
        #new_dataset = sistemas[0].copy()
        #new_lat     = sistemas[9].copy()
        #new_lon     = sistemas[10].copy()
        #new_topo    = sistemas[3].copy()
        #new_mascara = sistemas[5].copy()
        #new_f_field = sistemas[4].copy()
        #new_dx      = sistemas[16].copy()
        #new_dy      = sistemas[17].copy()
        
        new_dataset = transform_field_4d(new_dataset, rotate90, flip_vertical, flip_horizontal)
        new_lat     = transform_field_2d(new_lat    , rotate90, flip_vertical, flip_horizontal)
        new_lon     = transform_field_2d(new_lon    , rotate90, flip_vertical, flip_horizontal)
        new_topo    = transform_field_2d(new_topo   , rotate90, flip_vertical, flip_horizontal)
        new_mascara = transform_field_2d(new_mascara, rotate90, flip_vertical, flip_horizontal)
        
        # For the Coriolis f_field, apply the same spatial reordering...
        new_f_field = transform_field_2d(new_f_field, rotate90, flip_vertical, flip_horizontal)
        
        # For the grid spacing arrays dx and dy:
        new_dx = transform_field_2d(new_dx, rotate90, flip_vertical, flip_horizontal)
        new_dy = transform_field_2d(new_dy, rotate90, flip_vertical, flip_horizontal)
        if rotate90:
            # After a 90° rotation, the roles of dx and dy swap.
            new_dx, new_dy = new_dy, new_dx

    # -------------------------------
    # 2. Correct the velocity fields
    # -------------------------------
    # Assume velocity components are in dataset indices 1 (u) and 2 (v).
    # They have already been spatially re-ordered along with the dataset.
    # Now we apply the vector transformation.
    #
    # Define the elementary transformation matrices:
    # 90° counterclockwise rotation:
    T_rot = np.array([[0, 1],
                      [1,  0]])
    # Vertical flip (mirror in y): leaves u unchanged, reverses v.
    T_vflip = np.array([[1,  0],
                        [0, -1]])
    # Horizontal flip (mirror in x): reverses u, leaves v unchanged.
    T_hflip = np.array([[-1, 0],
                        [ 0, 1]])
    
    # Compose the overall transformation matrix.
    # (Operations are applied in the same order as above.)
    T = np.eye(2)
    if flip_vertical:
        T = T_vflip @ T
        new_dummy = copy.deepcopy(sistemas[20])
        if new_dummy:
            new_f_field = new_f_field*(-1)
    if flip_horizontal:
        T = T_hflip @ T
        new_dummy = copy.deepcopy(sistemas[20])
        if not new_dummy:
            new_f_field = new_f_field*(-1)
    if rotate90:
        T = T_rot @ T
        if sistemas[20]:
            new_dummy = 0
            new_f_field = new_f_field*(-1)
        else:
            new_dummy = 1

    # Extract the velocity components (they are 3D: [time, lat, lon]).
    u_field = new_dataset[1, :, :, :].copy()
    v_field = new_dataset[2, :, :, :].copy()
    
    # Apply the transformation matrix elementwise:
    # new_u = T[0,0]*u + T[0,1]*v, new_v = T[1,0]*u + T[1,1]*v.
    new_u_field = T[0,0] * u_field + T[0,1] * v_field
    new_v_field = T[1,0] * u_field + T[1,1] * v_field
    
    # Put the corrected velocities back into the dataset.
    new_dataset[1, :, :, :] = new_u_field
    new_dataset[2, :, :, :] = new_v_field

    return [new_dataset, sistemas[1], sistemas[2], new_topo,
            new_f_field,new_mascara, sistemas[6], sistemas[7],
            sistemas[8],new_lat,new_lon, sistemas[11],
            sistemas[12], sistemas[13],sistemas[14],sistemas[15],
            new_dx,new_dy,sistemas[18],sistemas[19],new_dummy,sistemas[21]
            ]
    
################################################################################################################################################



def Pega_Simulador_variables(PATH,croco_file,croco_blk_file,lista_hist,lista_verticais,
                             lista_blk,args,inicio=100,final=100,bulk_boolean=False,
                             Mean_Std=None,Mean_Std_bulk=None,time=10,zeta=True,TemCoriolis=0,
                             TemADV=0,slice_lat=slice(None),slice_lon=slice(None),domain='Closed'):
    #####################################################
    # Pega Arquivo netcdf e retorna Simulador_Variables #
    #####################################################
    
    system, bot, top, Topografia, Coriolis, mascara, lat, lon, dx,dy,dt,CoriolisDummy = Pega_Sistema3(PATH,croco_file,croco_blk_file,lista_hist,lista_verticais,lista_blk,
                                                                               inicio=inicio,final=final,bulk_boolean=bulk_boolean,
                                                                               Mean_Std=Mean_Std,Mean_Std_bulk=Mean_Std_bulk) # Normalização

    dx = dx.astype(np.float32)
    dy = dy.astype(np.float32)
    dt = dt.astype(np.float32)

    ## ANTIGO SISTEMA QUE CRIAVA NP.ARRAY PARA GUARDAR SISTEMA EM GRIDS IGUAIS COM PASDDING EM Ubar e Vba ##################
    #new_system = np.zeros((len(system),len(system[0]),system[0].shape[1],system[0].shape[2]), dtype = np.float32)
    #if bulk_boolean:
    #    new_top = np.zeros((len(top),len(top[1]),system[0].shape[1],system[0].shape[2]), dtype = np.float32) # top[0] é o bulk_dummy
    #else:
    #    new_top = copy.deepcopy(new_system) # []
    
    #for t in range( len(system) ): #variavel
    #    for tt in range( len(system[t]) ): # tempo
    #        try: #rho_i,j
    #            new_system[t,tt] = system[t][tt]
    #        except:
    #             if system[t][tt].shape[0] == system[0].shape[1]: # lat_vetor == lat_rho, len(lon_vetor) == len(lon_rho)-1
    #                 # u
    #                 new_system[t,tt,:,1:] = system[t][tt]
    #                 new_system[t,tt,:,0] = system[t][tt][:,0]
    #             else: #v_i+1/2,j  # len(lat_vetor) = len(lat_rho)-1
    #                 new_system[t,tt,1:,:] = system[t][tt]
    #                 new_system[t,tt,0] = system[t][tt][0]
    #
    #if bulk_boolean:
    #    new_top[0] = top[0]
    #    for t in range( 1,len(top) ): # tirando o bulk_dummy
    #        for tt in range( len(top[t]) ):
    #            try:
    #                new_top[t,tt] = top[t][tt] # rho...
    #            except:
    #                try: #u_i,j+1/2
    #                    new_top[t,tt,:,1:] = top[t][tt]
    #                except: #v_i+1/2,j
    #                    new_top[t,tt,1:,:] = top[t][tt]
    ########################################################################################################################
    
    for i in range(len(system)):
        system[i] = system[i][:, slice_lat[i], slice_lon[i]]
        # TOP FORCINGS NOT IMPLEMENTED YET
        #top[i] = top[i][:, slice_lat, slice_lon]
    
    # obs: o corte de [:,:,0:200,0:80] ([0:200,0:80] para vetores de 2 dim) é a costa do Brasil
    return [system, bot, top,
            Topografia[slice_lat[0],slice_lon[0]],Coriolis[slice_lat[0],slice_lon[0]],mascara[slice_lat[0],slice_lon[0]],
            args.n_variables,args.posicoes,args.posicoes_top,
            lat[slice_lat[0],slice_lon[0]],lon[slice_lat[0],slice_lon[0]],time,
            Mean_Std,zeta,TemCoriolis,TemADV, # era bulk_boolean...
            dx[slice_lat[0],slice_lon[0]],dy[slice_lat[0],slice_lon[0]],dt,domain,CoriolisDummy,bulk_boolean
           ]


def Pega_Sistema3(PATH,file,blk_file,lista_hist,lista_verticais,lista_bulk,
                inicio=8776,final=8000,bulk_boolean=True,Mean_Std=[],Mean_Std_bulk=[]):

    ##################################################################
     ### FUNÇÃO QUE PEGA OUTPUTS DO MODEL RUN DIRETO DOS ARQUIVOS ### 
      ##      NETCDF E DEVOLVE UMA LISTA                          ##
       #                          COM OS VETORES ACHATADOS.       #
        #       CADA ITEM DA LISTA É UMA variavel               #
          ######################################################
    if len(Mean_Std) >0:
        print( 'Inicializando sistema com normalização')
    else:
        print( 'Inicializando sistema sem normalização')
    
    CROCO_hist = xr.open_mfdataset(PATH+file)
    mascara=CROCO_hist['mask_rho']
        # tempo marcado em segundos... para passar para dias: /(3600*24)
    # Intervalo[inicio,fim] onde a simulação do ROMS ja esta estável.
    final += inicio
    i=-1 # contador de variaveis para cada corte
    
    # # # # # # # # # # # # # # #
    ### SYSTEMA E BOUNDARIES ####
    #############################
    
    sup = []
    bot = []
    for t in progressbar(range(len(lista_hist)), "Hist Variables: "):
        variable = lista_hist[t]
        
        #print(variable)
        i+=1 # contador de variaveis de cada corte
        
        if variable in lista_verticais: 
            # Só estou interessado na camada de superfície e na logo abaixo dela
            #print(inicio)
            Variavel = CROCO_hist[variable][inicio:final,:2].compute()
        
            #############
            # Superficie #
            #############
            
            temp=Variavel[:,0]
            
            # Achata, Normaliza e Concatena com as demais variaveis desse corte.
            vector=temp.values
            
            ################################
            # Camada inferior a superfície #
            ################################
            
            temp=Variavel[:,1]

            botom_vector = temp.values
            
            # ### NORMALIZAÇÃO ####
            #if len(Mean_Std) >0:
            #    if Mean_Std[1,t] > 0: # No começo da model run tem variaveis zeradas...
            #        botom_vector = (botom_vector-Mean_Std[0,t])/Mean_Std[1,t]
            #        vector =  vector/Mean_Std[1,t]# (vector-Mean_Std[0,t])/Mean_Std[1,t]
            bot.append(botom_vector) 
            sup.append(vector)
                
            
        else:
            # 2D Barotropicas
            Variavel = CROCO_hist[variable][inicio:final].compute() 
            
            ########################
            # Só existe na superfície #
            ########################
            temp=Variavel
            # Achata, Normaliza e Concatena com as demais variaveis desse corte.
            vector=temp.values
            print('numpy ndarray precision: ',vector.dtype)
            
            # ### NORMALIZAÇÃO ####
            #if len(Mean_Std) > 0:
            #    if Mean_Std[1,t] > 0: # No começo da model run tem variaveis zeradas...
            #        print('Normalizando Variavel', t)
            #        # Media 0, DESVIO PADRAO 1...
            #        vector = (vector-Mean_Std[0,t])/Mean_Std[1,t]#(vector-Mean_Std[0,t])/Mean_Std[1,t]
            sup.append(vector)
        
    # # # # # # # # # # # # >
    # Topografia E CORIOLIS ##
    # # # # # # # # # # # # <
    
    Topografia = CROCO_hist['h'].values
    # OBS: Até dominios sem UV_COR carregam matriz f não nula...
    # a nao influencia do termo na evolucao de Ubar e Vbar acontece no Step2D.F
    Coriolis = CROCO_hist['f'].values 
    CoriolisDummy = 0
    try:
        lat_rho = CROCO_hist['lat_rho'].values
        lon_rho = CROCO_hist['lon_rho'].values
    except:
        lat_rho = CROCO_hist['y_rho'].values
        lon_rho = CROCO_hist['x_rho'].values
    t+=1
    #if len(Mean_Std) > 0:
    #    if Mean_Std[1,t] > 0:
    #        print('Normalizando Topografia', t)
    #        # DESVIO PADRAO 1...
    #        Topografia = Topografia/Mean_Std[1,t]
        
    t+=1
    #if len(Mean_Std) > 0:
    #    if Mean_Std[1,t] > 0:
    #        print('Normalizando Coriolis', t)
    #        # DESVIO PADRAO normalizado igual ao momento...
    #        Coriolis = Coriolis/Mean_Std[1,t] #(Coriolis-Mean_Std[0,t])/Mean_Std[1,t]
    
    # DX
    t+=1
    dx = 1/CROCO_hist['pm'].values
    #if len(Mean_Std) > 0:
    #     if Mean_Std[1,t] > 0: # No começo da model run tem variaveis zeradas...
    #        print('Normalizando dx', t)
    #         # Media zero, DESVIO PADRAO 1...
    #        dx = (dx-Mean_Std[0,t])/Mean_Std[1,t]
    # DY
    t+=1
    dy = 1/CROCO_hist['pn'].values
    #if len(Mean_Std) > 0:
    #     if Mean_Std[1,t] > 0: # No começo da model run tem variaveis zeradas...
    #        print('Normalizando dy', t)
    #         # Media zero, DESVIO PADRAO 1...
    #        dy = (dy-Mean_Std[0,t])/Mean_Std[1,t]
    # DT
    t+=1
    # dt = em segundos!!!
    dt = (CROCO_hist['time'][1].values-CROCO_hist['time'][0].values)

    #if len(Mean_Std) > 0:
    #    if Mean_Std[1,t] > 0: # No começo da model run tem variaveis zeradas...
    #        print('Normalizando t', Mean_Std[1,t], t)
    #        # Media zero, DESVIO PADRAO 1...
    #        dt = (dt-Mean_Std[0,t])/Mean_Std[1,t]

    
    ###### #### #### ####
    ### BULK FORCING ####
    ###### #### #### #### 
    if bulk_boolean:
        if blk_file == '':
            
            bulk = []
            bulk.append(0) # Bulk dummy
            
            # Cria matriz de dimensoes iguais as rho de valores zerados para cada item da lista_bulk
            for t in progressbar(range(1,len(lista_bulk)), "Bulk Variables: "): # começa do 1 (sem o bulk_dummy)
                #variable = lista_bulk[t]
                VETOR = np.zeros(sup[0].shape) # sup[0] é o zeta...
                bulk.append(VETOR)
            
        else:
            CROCO_blk = xr.open_mfdataset(PATH+blk_file) # Atmosphere/heatflux forcing at ocean surface

            # TEMPOS PARA AVALIAR QUAL FOI A FORÇANTE USADA #
            time = CROCO_hist['time'][inicio:final].values # esta em segundos
            
            lenght = len(time)
            
            TIME=CROCO_blk['bulk_time'].values 
            # esta em nano segundos por alguma razao incrompreensível.
            # para passar para segundos: /1000000000
    
            TIME=TIME.astype('float64')/1e9
            tempo = np.zeros(lenght)
            ## Cria identificador do tempo forçante utilizado a cada passo do modelo ##
            counter=0 
            
            for t in range(lenght):
                if time[t]>=TIME[counter]: # o tempo da simução atingiu os bulk times
                    counter+=1
                    tempo[t:] = counter
                if counter == 11: # passou de 1 ano, logo tenho que resetar o BULK... acho...
                    TIME += TIME[11]
                    counter=0
                    
            
    
            j=-1 # contador de variaveis para o BULK
            bulk = []
            bulk.append(1) # Bulk dummy
            for t in progressbar(range(1,len(lista_bulk)), "Bulk Variables: "): # começa do 1 (sem bulk_dummy)
                variable = lista_bulk[t]
                
                j+=1 # contador de variaveis de cada corte
                Variavel=CROCO_blk[variable] 
                # ____________ ##
                # NORMALIZACAO ##
                # ____________ ##
                
                temp=Variavel 
                vector=temp.values
                
                # pega a forçante usada em cada tempo da simulação
                VETOR = np.zeros((len(tempo),vector.shape[1],vector.shape[2])) 
                for tem in range(len(tempo)):
                    VETOR[tem]=vector[int(tempo[tem])]
                # Normalização 
                #if len(Mean_Std_bulk) > 0:
                #    if Mean_Std_bulk[1,t] > 0: # No começo da model run tem variaveis zeradas...
                #        VETOR = VETOR/Mean_Std_bulk[1,t]  # STD ->1
                #        #VETOR = (VETOR-Mean_Std_bulk[0,t])/Mean_Std_bulk[1,t]  # MEAN ->0 e STD ->1
                bulk.append(VETOR)
    else:
        bulk =[]
    
    return sup, bot, bulk, Topografia, Coriolis, mascara, lat_rho, lon_rho, dx, dy, dt, CoriolisDummy


def Triple_Plot(lat,lon,
                plot1,plot2,plot3,
                title1='Truth',title2='Model prediction',title3='Bias',
                levels=[],save=False,folder=None,dominio='ocean',Energy=False):
        #plot figure
        # dominio = 'ocean' ou 'abstract'

        if len(levels) == 0:
            MAXIMO, MINIMO = np.ndarray.max(plot1),np.ndarray.min(plot1)
            limites = np.max([MAXIMO,-MINIMO])
            levels = np.linspace(-limites, limites, 10)
            
        if dominio == 'ocean':
            projection=ccrs.PlateCarree(central_longitude=-90.0, globe=None)
            transform=ccrs.PlateCarree()
        else:
            projection='rectilinear'
            transform=None
        
        CMAP = "PuOr_r"
        if Energy:
            CMAP = 'Purples'
        
        fig = plt.figure(figsize=(10,4))
        
        ax = fig.add_subplot(1, 3, 1,
                            aspect='equal',
                            projection=projection)   
        
        im = ax.contourf(lon, lat, plot1,
                         cmap=CMAP,
                         levels=levels,
                         extend='both',transform=transform)
        
        ax.set_title(str(title1))# Passo: ' + str(time) )
        if dominio == 'ocean':
            ax.coastlines(linewidths=1)
            ax.add_feature(cf.BORDERS, linestyle=':', alpha=.5)
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                             linewidth=1.2, color='gray', alpha=0.5, linestyle=':')
            gl.top_labels = False
            gl.bottom_labels = False
            gl.right_labels = False
        cbar1 = plt.colorbar(im, pad=0.1, aspect=20,shrink=0.4,location='right')
        
        
        ax= fig.add_subplot(1, 3, 2,
                            aspect='equal',
                            projection=projection)   
        im = ax.contourf(lon, lat, plot2,
                         cmap=CMAP,
                         levels=levels,
                         extend='both',transform=transform)
        ax.set_title(str(title2))# Passo: ' + str(time) )
        if dominio == 'ocean':
            ax.coastlines(linewidths=1)
            ax.add_feature(cf.BORDERS, linestyle=':', alpha=.5)
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                             linewidth=1.2, color='gray', alpha=0.5, linestyle=':')
            gl.top_labels = False
            gl.bottom_labels = False
            gl.right_labels = False
        cbar2 = plt.colorbar(im, pad=0.1, aspect=20,shrink=0.4,location='right')

        # NEW LEVELS FOR SMALLER BIAS
        levels = levels #/ 10
        #with np.printoptions(threshold=np.inf):
        #    print(plot3)
        
        ax= fig.add_subplot(1, 3, 3,
                            aspect='equal',
                            projection=projection)   
        im = ax.contourf(lon, lat, plot3,
                         cmap="PuOr_r",
                         levels=levels,
                         extend='both',transform=transform)
        ax.set_title(str(title3)) # Passo: ' + str(time) )
        if dominio == 'ocean':
            ax.coastlines(linewidths=1)
            ax.add_feature(cf.BORDERS, linestyle=':', alpha=.5)
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                              linewidth=1.2, color='gray', alpha=0.5, linestyle=':')
            gl.top_labels = False
            gl.bottom_labels = False
            gl.right_labels = False
        cbar3 = plt.colorbar(im, pad=0.1, aspect=20,shrink=0.4,location='right')

        # Cbar
        formatter = FuncFormatter(sci_formatter)
        cbar1.ax.yaxis.set_major_formatter(formatter)
        cbar2.ax.yaxis.set_major_formatter(formatter)
        cbar3.ax.yaxis.set_major_formatter(formatter)

        plt.tight_layout()
        
        if save:
            # saving the figure. 
            plt.savefig(folder)
            plt.close()
        else:
            plt.show()  
            
def One_Plot(lat,lon,plot1,
              levels=[],save=False,folder=None,dominio='ocean',Energy=False):
        #plot1 is one model, plot 2 is second model, plot 3 is bias
        #plot1[0], plot2[0] and plot3[0] are ssh
        #plot1[0], plot2[0] and plot3[0] are vorticity
        # dominio = 'ocean' ou 'abstract'
        if len(levels) == 0:
            MAXIMO, MINIMO = np.ndarray.max(plot1[0]),np.ndarray.min(plot1[0])
            limites = np.max([MAXIMO,-MINIMO])
            MAXIMO2, MINIMO2 = np.ndarray.max(plot1[1]),np.ndarray.min(plot1[1])
            limites2 = np.max([MAXIMO2,-MINIMO2])
            levels = [np.linspace(-limites, limites, 10),np.linspace(-limites2, limites2, 10)]
            
        if dominio == 'ocean':
            projection=ccrs.PlateCarree(central_longitude=-90.0, globe=None)
            transform=ccrs.PlateCarree()
        else:
            projection='rectilinear'
            transform=None
        
        CMAP = "PuOr_r"
        if Energy:
            CMAP = 'Purples'
        
        fig = plt.figure(figsize=(7,4))

       # Primeiro modelo, ssh plot1[0] e vorticity plot1[1]
        ax = fig.add_subplot(1, 2, 1,
                            aspect='equal',
                            projection=projection)   
        im = ax.contourf(lon, lat, plot1[0],
                         cmap=CMAP,
                         levels=levels[0],
                         extend='both',transform=transform)
        ax.set_title('sea surface height')
        if dominio == 'ocean':
            ax.coastlines(linewidths=1)
            ax.add_feature(cf.BORDERS, linestyle=':', alpha=.5)
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                             linewidth=1.2, color='gray', alpha=0.5, linestyle=':')
            gl.top_labels = False
            gl.bottom_labels = False
            gl.right_labels = False
        # CB    
        #cbar1 = plt.colorbar(im, pad=0.1, aspect=20,shrink=0.4,location='left')
        
        ax= fig.add_subplot(1, 2, 2,
                            aspect='equal',
                            projection=projection)   
        im = ax.contourf(lon, lat, plot1[1],
                         cmap=CMAP,
                         levels=levels[1],
                         extend='both',transform=transform)
        ax.set_title('vorticity')
        ax.set_yticklabels([])
        if dominio == 'ocean':
            ax.coastlines(linewidths=1)
            ax.add_feature(cf.BORDERS, linestyle=':', alpha=.5)
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                             linewidth=1.2, color='gray', alpha=0.5, linestyle=':')
            gl.top_labels = False
            gl.bottom_labels = False
            gl.right_labels = False
        # CB    
        #cbar2 = plt.colorbar(im, pad=0.1, aspect=20,shrink=0.4,location='right')

        plt.tight_layout()
        
        if save:
            # saving the figure. 
            plt.savefig(folder)
            plt.close()
        else:
            plt.show()  

def Bias_Plot(lat,lon,plot1,plot2,plot3,
              levels=[],save=False,folder=None,dominio='ocean',Energy=False):
        #plot1 is one model, plot 2 is second model, plot 3 is bias
        #plot1[0], plot2[0] and plot3[0] are ssh
        #plot1[0], plot2[0] and plot3[0] are vorticity
        # dominio = 'ocean' ou 'abstract'
        if len(levels) == 0:
            MAXIMO, MINIMO = np.ndarray.max(plot1[0]),np.ndarray.min(plot1[0])
            limites = np.max([MAXIMO,-MINIMO])
            MAXIMO2, MINIMO2 = np.ndarray.max(plot1[1]),np.ndarray.min(plot1[1])
            limites2 = np.max([MAXIMO2,-MINIMO2])
            levels = [np.linspace(-limites, limites, 10),np.linspace(-limites2, limites2, 10)]
            
        if dominio == 'ocean':
            projection=ccrs.PlateCarree(central_longitude=-90.0, globe=None)
            transform=ccrs.PlateCarree()
        else:
            projection='rectilinear'
            transform=None
        
        CMAP = "PuOr_r"
        if Energy:
            CMAP = 'Purples'
        
        fig = plt.figure(figsize=(7,12))

       # Primeiro modelo, ssh plot1[0] e vorticity plot1[1]
        ax = fig.add_subplot(3, 2, 1,
                            aspect='equal',
                            projection=projection)   
        im = ax.contourf(lon, lat, plot1[0],
                         cmap=CMAP,
                         levels=levels[0],
                         extend='both',transform=transform)
        ax.set_title('Model 1 sea surface height')
        if dominio == 'ocean':
            ax.coastlines(linewidths=1)
            ax.add_feature(cf.BORDERS, linestyle=':', alpha=.5)
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                             linewidth=1.2, color='gray', alpha=0.5, linestyle=':')
            gl.top_labels = False
            gl.bottom_labels = False
            gl.right_labels = False
        # CB    
        #cbar1 = plt.colorbar(im, pad=0.1, aspect=20,shrink=0.4,location='left')
        
        ax= fig.add_subplot(3, 2, 2,
                            aspect='equal',
                            projection=projection)   
        im = ax.contourf(lon, lat, plot1[1],
                         cmap=CMAP,
                         levels=levels[1],
                         extend='both',transform=transform)
        ax.set_title('Model 1 vorticity')
        ax.set_yticklabels([])
        if dominio == 'ocean':
            ax.coastlines(linewidths=1)
            ax.add_feature(cf.BORDERS, linestyle=':', alpha=.5)
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                             linewidth=1.2, color='gray', alpha=0.5, linestyle=':')
            gl.top_labels = False
            gl.bottom_labels = False
            gl.right_labels = False
        # CB    
        #cbar2 = plt.colorbar(im, pad=0.1, aspect=20,shrink=0.4,location='right')


        # Primeiro modelo, ssh plot1[0] e vorticity plot1[1]
        ax = fig.add_subplot(3, 2, 3,
                            aspect='equal',
                            projection=projection)   
        
        im = ax.contourf(lon, lat, plot2[0],
                         cmap=CMAP,
                         levels=levels[0],
                         extend='both',transform=transform)
        
        ax.set_title('Models 2 sea surface height')
    
        if dominio == 'ocean':
            ax.coastlines(linewidths=1)
            ax.add_feature(cf.BORDERS, linestyle=':', alpha=.5)
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                             linewidth=1.2, color='gray', alpha=0.5, linestyle=':')
            gl.top_labels = False
            gl.bottom_labels = False
            gl.right_labels = False
        # CB    
        #cbar1 = plt.colorbar(im, pad=0.1, aspect=20,shrink=0.4,location='left')
        
        
        ax= fig.add_subplot(3, 2, 4,
                            aspect='equal',
                            projection=projection)   
        im = ax.contourf(lon, lat, plot2[1],
                         cmap=CMAP,
                         levels=levels[1],
                         extend='both',transform=transform)
        ax.set_title('Model 2 vorticity')
        ax.set_yticklabels([])
        if dominio == 'ocean':
            ax.coastlines(linewidths=1)
            ax.add_feature(cf.BORDERS, linestyle=':', alpha=.5)
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                             linewidth=1.2, color='gray', alpha=0.5, linestyle=':')
            gl.top_labels = False
            gl.bottom_labels = False
            gl.right_labels = False
        # CB    
        #cbar2 = plt.colorbar(im, pad=0.1, aspect=20,shrink=0.4,location='right')
    
        # NEW LEVELS FOR BIAS
        limite = np.max([levels[0],-levels[0]])
        levels[0] = np.linspace(-limite, limite, 50)
        limite = np.max([levels[1],-levels[1]])
        levels[1] = np.linspace(-limite, limite, 50)
    
        #with np.printoptions(threshold=np.inf):
        #    print(plot3)
        # Primeiro modelo, ssh plot1[0] e vorticity plot1[1]
        ax = fig.add_subplot(3, 2, 5,
                            aspect='equal',
                            projection=projection)   
        
        im = ax.contourf(lon, lat, plot3[0],
                         cmap=CMAP,
                         levels=levels[0],
                         extend='both',transform=transform)
        
        ax.set_title('Bias sea surface height')
    
        if dominio == 'ocean':
            ax.coastlines(linewidths=1)
            ax.add_feature(cf.BORDERS, linestyle=':', alpha=.5)
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                             linewidth=1.2, color='gray', alpha=0.5, linestyle=':')
            gl.top_labels = False
            gl.bottom_labels = False
            gl.right_labels = False
       
        ## CB    
        #cbar1 = plt.colorbar(im, pad=0.1, aspect=20,shrink=0.4,location='left')
        
        
        ax= fig.add_subplot(3, 2, 6,
                            aspect='equal',
                            projection=projection)   
        im = ax.contourf(lon, lat, plot3[1],
                         cmap=CMAP,
                         levels=levels[1],
                         extend='both',transform=transform)
        ax.set_title('Bias vorticity')
        ax.set_yticklabels([])
        if dominio == 'ocean':
            ax.coastlines(linewidths=1)
            ax.add_feature(cf.BORDERS, linestyle=':', alpha=.5)
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                             linewidth=1.2, color='gray', alpha=0.5, linestyle=':')
            gl.top_labels = False
            gl.bottom_labels = False
            gl.right_labels = False
        # CB    
        #cbar2 = plt.colorbar(im, pad=0.1, aspect=20,shrink=0.4,location='right')

        # CB    
        #formatter = FuncFormatter(sci_formatter)
        #cbar1.ax.yaxis.set_major_formatter(formatter)
        #cbar2.ax.yaxis.set_major_formatter(formatter)

        plt.tight_layout()
        
        if save:
            # saving the figure. 
            plt.savefig(folder)
            plt.close()
        else:
            plt.show()  

def SincNetwork(m,dt):
    # Changes the network weights to adjust to new temporal res #
    with torch.no_grad():
        m.bilinear2.weight[0, 0, 1] = -dt
        m.bilinear2.weight[0, 1, 1] = dt
        m.bilinear2.weight[0, 2, 3] = -dt
        m.bilinear2.weight[0, 3, 3] = dt
            