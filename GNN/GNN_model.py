import torch
import torch.nn as nn


class SchNet(nn.Module):
    def __init__(self, feature_number, element_number, filter_number, layer_number, pooling, ghost_killer=True):
        super(SchNet, self).__init__()
        self.ghost_killer=ghost_killer
        self.pooling=pooling # type of pooling at the end, support "sum", "first" or "mean"
        self.filter_number=filter_number # the dim of filter for mapping the distances
        self.element_number=element_number # 
        self.feature_number=feature_number # the dim of atom embedding 
        self.layer_number=layer_number # the no of interaction layer
        self.encoding=Encoding(self.feature_number,self.element_number, ghost_killer) # initiation of the atoms' embedding 
        # passing the embedded atoms to interaction block for layer_number times 
        self.interaction_layers=nn.ModuleList([interaction(self.feature_number, self.filter_number, ghost_killer) for i in range(self.layer_number)]) 
        self.second_last_layer=AtomWise(self.feature_number,output_feature_number=int(self.feature_number/2)) # the first atom_wise layer in main arch 
        #self.ssp=SSP()
        self.ssp=torch.nn.ELU() 
        self.final_layer=AtomWise(int(self.feature_number/2), output_feature_number=1) # the 2nd atom_wise layer of the main arch
    def forward(self,elements,filters, atom_number, mask=None):
        #Input dimensions elements: (N*A) scalar then?! 
        #Input dimensions filters: (N,A,A,R)
        
        x=self.encoding(elements) # output dim: (N*A,F)
        
        for i in range(self.layer_number):
            x=self.interaction_layers[i](x, filters, atom_number) # ATOM_NUMBER IS REDUNDANT
        
        x=self.second_last_layer(x, atom_number) 
        
        x=self.ssp(x)
        x=self.final_layer(x, atom_number)
        x=torch.reshape(x,(-1,atom_number))
        if not mask==None:
            x=x*mask
        if self.pooling=="mean":
            if not mask==None:
                x=torch.div(torch.sum(x,-1),torch.sum(mask,-1))
            else:
                x=torch.sum(x,-1)/atom_number # evaluating a mean value over the nomber of atoms in the mask
        elif self.pooling=="sum":
            x=torch.sum(x,-1) # evaluating a mean value over the nomber of atoms 
        elif self.pooling=="first":
            x=x[:,0]
        #Output dimensions: (N)
        return(x)
        
class Encoding(nn.Module):
    '''
    Recieves the list of elements to generate a radom vector used as embedding.
    The embedding has a dim of F aka feature_number.
    '''
    def __init__(self, feature_number,element_number, ghost_killer):
        super(Encoding, self).__init__()
        self.feature_number=feature_number
        self.element_number=element_number
        self.ghost_killer=ghost_killer
        optimized=True
        if optimized:
            self.encodings = torch.nn.Parameter(torch.randn(self.element_number,self.feature_number))
            self.encodings.requires_grad = True      
        else:
            lists=[[i]*self.feature_number for i in range(element_number)]
            self.encodings=torch.tensor(lists,dtype=torch.float32)
    def forward(self,element_list):
        #incoming dimensions: (N,A)
        element_list=element_list.reshape(-1)
        if not self.ghost_killer:
            vectors=[self.encodings[element] for element in element_list]
        else:
            vectors=[self.encodings[element] if not element==0 else torch.tensor([0]*self.feature_number) for element in element_list]
        result=torch.reshape(torch.stack(vectors),(-1,self.feature_number)) 
        #Outgoing dimensions: (N*A,F)
        return result

class AtomWise(nn.Module):
    '''
    Basically a linear layer to reduce the last dim of input.
    It does not need a atom_no to work with!
    '''
    def __init__(self, feature_number, output_feature_number=None):
        super(AtomWise, self).__init__()
        if output_feature_number==None:
            self.output_feature_number=feature_number
        else:
            self.output_feature_number=output_feature_number
        self.feature_number=feature_number
        self.layer_1=nn.Linear(self.feature_number, self.output_feature_number)

    def forward(self,x, atom_number): # WHAT IS ATOM NO, IT IS NOT EVEN USED!
        #Input dimension: (N*A,F)
        result=self.layer_1(x) 
        #output dimension: (N*A,F)
        return result

class SSP(nn.Module):
    def forward(self, x):
        return torch.log(0.5*torch.exp(x)+0.5)

class cfconv(nn.Module):
    '''
    The code block's responsible for updating the atom embeddings using coordinates and distances through filters.  
    '''
    def __init__(self, feature_number, filter_number,ghost_killer):
        super(cfconv, self).__init__()
        self.filter_number=filter_number
        self.feature_number=feature_number
        if ghost_killer:
            self.layer_1=nn.Linear(self.filter_number,self.feature_number, bias=False)
        else:
            self.layer_1=nn.Linear(self.filter_number,self.feature_number)
        #self.ssp=SSP()
        self.ssp=torch.nn.ELU()
        #self.layer_2=nn.Linear(self.feature_number,self.feature_number)
         
    def forward(self,x,filter_inputs, atom_number):
        #Input dimension x: (N*A,F)
        #Input dimension filter_inputs: (N,A,A,R)
        x=torch.reshape(x,(-1,atom_number, self.feature_number)) # dim: (N,A,F)
        filter_inputs=torch.reshape(filter_inputs,(-1,self.filter_number))
        y=self.layer_1(filter_inputs) # dim: (N,A,A,F)
        y=self.ssp(y)
        filters=torch.reshape(y,(-1,atom_number, atom_number, self.feature_number)) # dim: (N,A,A,F)
        product=x.unsqueeze(2)*filters # (N,A,1,F)*(N,A,A,F) element_wise multiplication -> output dim (N,A,A,F)
        result=torch.reshape(torch.sum(product,-2),(-1,self.feature_number)) # SHOULDN'T SUM BE ON DIM=-1 INSTEAD? 
        #Output dimension: (N*A,F)
        return(result)
    
class interaction(nn.Module):
    '''
    Interaction block further updates the embeddings using cfconv and multiple atom_wise layers.
    '''
    def __init__(self,feature_number, filter_number,ghost_killer):
        super(interaction, self).__init__()
        self.filter_number=filter_number
        self.feature_number=feature_number
        self.atomwise1=AtomWise(self.feature_number)
        self.atomwise2=AtomWise(self.feature_number)
        self.atomwise3=AtomWise(self.feature_number)
        self.cfconv1=cfconv(self.feature_number, self.filter_number,ghost_killer)
        #self.ssp=SSP()
        self.ssp=torch.nn.ELU()

    def forward(self, x, filter_inputs, atom_number):
        #Input dimension x: (N*A,F)
        #Input dimension filter_inputs: (N,A,A,R)
        v=self.atomwise1(x,atom_number) # NO OF ATOMS IS A REDUNDANT PARAM!
        v=self.cfconv1(v,filter_inputs,atom_number)  # NO OF ATOMS IS ONLY FOR FIXING THE DIM OF MATRICES. 
        v=self.atomwise2(v,atom_number)
        v=self.ssp(v)
        v=self.atomwise3(v,atom_number)
        output=x+v
        #Output dimension: (N*A,F)
        return(output)