import torch

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):

        # if self.kwargs['use_weights']:
        #     threshold = 0.05
        #     s_hyper = 2.0
        #     trunc_index = inputs > threshold
            
        #     weights = torch.ones_like(inputs)
        #     weights[trunc_index] = torch.exp( -1.0  * (inputs[trunc_index] - threshold)**2 / (2* s_hyper **2))

        #     return torch.cat([weights * fn(inputs) for fn in self.embed_fns], -1)

        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, input_dims = 3, i=0,include_input = True, use_weights = False):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : include_input,
                'use_weights' : use_weights,
                'input_dims' : input_dims,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim

# Positional encoding
class Trigonometric_kernel:
    def __init__(self, L = 10, input_dims = 3, include_input=True, use_weights = False):

        self.L = L
 
        self.embed_fn, self.out_ch= get_embedder(L, input_dims, include_input = include_input, use_weights = use_weights)

    '''
    INPUT
     x: input vectors (N,C) 

     OUTPUT

     pos_kernel: (N, calc_dim(C) )
    '''
    def __call__(self, x):
        return self.embed_fn(x)

    def calc_dim(self, dims=0):
        return self.out_ch