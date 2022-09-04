import torch
import torch.nn as nn
import math


class base_Conv_net(nn.Module):
    def __init__(self, k_size ):
        super(base_Conv_net, self).__init__()

        self.conv1d_1 = nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = k_size, padding = int((k_size-1)/2))   
        self.batch_norm_1 = nn.BatchNorm1d(32)

        self.conv1d_2 = nn.Conv1d(in_channels = 32, out_channels = 16, kernel_size = k_size, padding = int((k_size-1)/2))   
        self.batch_norm_2 = nn.BatchNorm1d(16)

        self.conv1d_3 = nn.Conv1d(in_channels = 16, out_channels = 32, kernel_size = k_size, padding = int((k_size-1)/2))   
        self.batch_norm_3 = nn.BatchNorm1d(32)

    def forward(self, x):            
        ori_x = x
        x = torch.relu( self.batch_norm_1( self.conv1d_1( x ) ) )
        x = torch.relu( self.batch_norm_2( self.conv1d_2( x ) ) )
        x = torch.relu( self.batch_norm_3( self.conv1d_3( x ) ) )

        return x + ori_x

class BasicBlockall(nn.Module):
    def __init__(self,):
        super(BasicBlockall, self).__init__()

        self.bblock_1 = nn.ModuleList()
        for _ in range(2):
            self.bblock_1.append( base_Conv_net( 3  ) )

        self.bblock_2 = nn.ModuleList()
        for _ in range(2):
            self.bblock_2.append( base_Conv_net( 5 ) )

        self.bblock_3 = nn.ModuleList()
        for _ in range(2):
            self.bblock_3.append( base_Conv_net( 7 ) )


    def forward(self, inputs):
    
        out3 = self.bblock_1[1]( self.bblock_1[0]( inputs ) )

        out5 = self.bblock_2[1]( self.bblock_2[0]( inputs ) )

        out7 = self.bblock_3[1]( self.bblock_3[0]( inputs ) )

        out = torch.cat( [out3,out5,out7] , dim = 1 )
        
        return out

class Res_1d_cnn(nn.Module):
    def __init__(self, input_dim, T, DEVICE= torch.device('cuda:0')):
        super(Res_1d_cnn, self).__init__()
        self.ori_T = T

        self.input_dim = input_dim

        self.conv1 = nn.Conv1d(in_channels = input_dim, out_channels = 32, kernel_size = 5, padding = int((5-1)/2))   
        self.norm1 = nn.BatchNorm1d( 32 )
        
        self.con_block = BasicBlockall()

        self.conv2 =  nn.Conv1d(in_channels = 32*3, out_channels = 32, kernel_size = 1)   
        self.norm2 = nn.BatchNorm1d( 32 )

        self.conv3 =  nn.Conv1d(in_channels = 32, out_channels = input_dim, kernel_size = 1, padding = int((1-1)/2))   

        self.to(DEVICE) 

    def forward(self, x):
        
        x = self.conv1( x )
        x = torch.relu( self.norm1( x ) )

        x = self.con_block( x )
        x = torch.relu( self.norm2( self.conv2( x ) ) )

        x = self.conv3( x )
        
        return x

def make_model(args, in_channels, DEVICE):

    model = Res_1d_cnn(
        input_dim = in_channels, 
        T = 3000, 
        DEVICE = DEVICE
        )
    
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model 
