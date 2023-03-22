from typing import Any, Callable, Sequence, Optional, Union
from flax.core import freeze, unfreeze
import flax.linen as nn

import jax
import jax.numpy as jnp

Array = Any

class AsinhLayer(nn.Module):
    b_start: float = 0.005
    bias_init: Callable = nn.initializers.zeros
    a_init: Callable = nn.initializers.ones
    b_init: Callable = nn.initializers.ones
    c_init: Callable = nn.initializers.zeros
    d_init: Callable = nn.initializers.zeros
    
    @nn.compact
    def __call__(self, inputs):

        a = self.param('a', self.a_init, (1,))
        b = self.param('b', self.b_init, (1,))
        c = self.param('c', self.c_init, (1,))
        d = self.param('d', self.d_init, (1,)) 

        y = a*jnp.arcsinh(b*inputs*(1./self.b_start) + c) + d
        return y


class InceptBlock(nn.Module):
    """Inception block submodule"""
    filters: Sequence[int]
    strides: Union[None, int, Sequence[int]]
    dims: int
    do_5x5: bool = True
    do_3x3: bool = True
    #input_shape: Sequence[int]

    @nn.compact
    def __call__(self, x):

        outs = []
        
        if self.do_5x5:
        # 5x5 filter
          x1 = nn.Conv(features=self.filters[0], kernel_size=(1,)*self.dims, strides=None)(x)
          x1 = nn.Conv(features=self.filters[0], kernel_size=(5,)*self.dims, strides=self.strides)(x1)
          outs.append(x1)
          
        if self.do_3x3:
        # 3x3 filter
          x2 = nn.Conv(features=self.filters[1], kernel_size=(1,)*self.dims, strides=None)(x)
          x2 = nn.Conv(features=self.filters[1], kernel_size=(3,)*self.dims, strides=self.strides)(x2)
          outs.append(x2)

        # 1x1
        x3 = nn.Conv(features=self.filters[2], kernel_size=(1,)*self.dims, strides=None)(x)
        x3 = nn.Conv(features=self.filters[2], kernel_size=(1,)*self.dims, strides=self.strides)(x3)
        outs.append(x3)
        
        # maxpool and avgpool
        x4 = nn.max_pool(x, (3,)*self.dims, padding='SAME')
        x4 = nn.Conv(features=self.filters[3], kernel_size=(1,)*self.dims, strides=self.strides)(x4)
        outs.append(x4)
                    
        x = jnp.concatenate(outs, axis=-1)
        
        return x     
    

class InceptResBlock(nn.Module):
    """Inception Res block submodule"""
    filters: Sequence[int]
    strides: Union[None, int, Sequence[int]]
    dims: int
    do_5x5: bool = True
    do_3x3: bool = True
    #input_shape: Sequence[int]

    @nn.compact
    def __call__(self, x):

        outs = []

        xr = nn.Conv(features=self.filters[0], kernel_size=(1,)*self.dims, strides=self.strides)(x)
        
        if self.do_5x5:
        # 5x5 filter
          x1 = nn.Conv(features=self.filters[0], kernel_size=(1,)*self.dims, strides=None)(x)
          x1 = nn.Conv(features=self.filters[0], kernel_size=(5,)*self.dims, strides=self.strides)(x1)
          outs.append(xr+x1)
          
        if self.do_3x3:
        # 3x3 filter
          x2 = nn.Conv(features=self.filters[1], kernel_size=(1,)*self.dims, strides=None)(x)
          x2 = nn.Conv(features=self.filters[1], kernel_size=(3,)*self.dims, strides=self.strides)(x2)
          outs.append(xr+x2)

        # 1x1
        x3 = nn.Conv(features=self.filters[2], kernel_size=(1,)*self.dims, strides=None)(x)
        x3 = nn.Conv(features=self.filters[2], kernel_size=(1,)*self.dims, strides=self.strides)(x3)
        outs.append(xr+x3)
        
        # maxpool or avg_pool
        x4 = nn.max_pool(x, (3,)*self.dims, padding='SAME')
        x4 = nn.Conv(features=self.filters[3], kernel_size=(1,)*self.dims, strides=self.strides)(x4)
        outs.append(xr+x4)
                    
        x = jnp.concatenate(outs, axis=-1)
        
        return x     


#filters = (55,55,55,55)
#filters = (16,16,16,16)
filters = (16,16,16,16)

class InceptNet(nn.Module):
    """An incept net architecture"""
    
    @nn.compact
    def __call__(self, x):
        dim = 2
        x = InceptBlock(filters, strides=2, dims=2)(x) # 32
        x = nn.elu(x)
        x = InceptBlock(filters, strides=2, dims=2)(x) # 16
        x = nn.elu(x)
        x = InceptBlock(filters, strides=2, dims=2)(x) # 8
        x = nn.elu(x)
        x = InceptBlock(filters, strides=2, dims=2)(x) # 4
        x = nn.elu(x)
        x = InceptBlock(filters, strides=2, dims=2)(x) # 2
        x = nn.elu(x)
        x = InceptBlock(filters, strides=2, dims=2)(x) # 1
        x = nn.elu(x)
        x = nn.Conv(features=n_params, kernel_size=(1,)*dim, strides=None)(x)
        x = x.reshape(-1)
        return x


class InceptResNet(nn.Module):
    """An incept net architecture"""
    
    @nn.compact
    def __call__(self, x):
        dim = 2
        #x = AsinhLayer()(x)
        x = InceptResBlock(filters, strides=2, dims=2)(x)
        x = nn.elu(x)
        x = InceptResBlock(filters, strides=2, dims=2)(x)
        x = nn.elu(x)
        x = InceptResBlock(filters, strides=2, dims=2)(x)
        x = nn.elu(x)
        x = InceptResBlock(filters, strides=2, dims=2, do_5x5=False)(x) # 4
        x = nn.elu(x)
        x = InceptResBlock(filters, strides=2, dims=2, do_5x5=False)(x) # 2
        x = nn.elu(x)
        x = InceptResBlock(filters, strides=2, dims=2, do_5x5=False, do_3x3=False)(x) # 1
        x = nn.elu(x)
        x = nn.Conv(features=n_params, kernel_size=(1,)*dim, strides=None)(x)
        x = x.reshape(-1)
        return x



# (4,64,64,2)
# (4,32,32,2)
# (4,16,16,2)
# (4,8,8,2)
# (4,4,4,2)
# (2,2,2,2)
# (2,1,1,1)
# (2,) <- everything that’s translationally invariant is squished into two numbers

# custom activation function
@jax.jit
def almost_leaky(x: Array) -> Array:
  r"""Almost Leaky rectified linear unit activation function.

  Computes the element-wise function:

  .. math::
    \mathrm{almost\_leaky}(x) = \begin{cases}
      x, & x \leq -1\\
      - |x|^3/3, & -1 \leq x < 1\\
      3x & x > 1
    \end{cases}

  Args:
    x : input array
  """
  return jnp.where(x < -1, x, jnp.where((x < 1), ((-(jnp.abs(x)**3) / 3) + x*(x+2) + (1/3)), 3*x))



class CNNRes3D(nn.Module):
    """An incept net architecture"""
    filters: Sequence[int] = (10,10,10,10) #(5,5,5,5) #(2,2,2,2)
    div_factor: float = 0.005
    out_shape: int = 2
    do_big_convs: bool = True
    act: str = "almost_leaky"
    
    @nn.compact
    def __call__(self, x):
        fs = self.filters
        dbg = self.do_big_convs
        if self.act == "almost_leaky":
           act = almost_leaky
        else:
           act = nn.elu

        x /= self.div_factor
        x = InceptBlock(fs, strides=(1,2,2), dims=3)(x) # out: 4, 32, 32, 2
        x = act(x)
        x = InceptBlock(fs, strides=(1,2,2), dims=3)(x) # out: 4, 16, 16, 2
        x = act(x)
        x = InceptBlock(fs, strides=(1,2,2), dims=3)(x) # out: 4, 8, 8, 2
        x = act(x)
        x = InceptBlock(fs, strides=(1,2,2), dims=3)(x) # out: 4, 4, 4, 2
        x = act(x)
        x = InceptBlock(fs, strides=(2,2,2), dims=3, do_5x5=dbg)(x) # out: 2, 2, 2, 2
        x = act(x)
        x = InceptBlock(fs, strides=(2,2,2), dims=3, do_5x5=dbg, do_3x3=dbg)(x) # out: 1, 1, 1, 2
        x = act(x)
        x = nn.Conv(self.out_shape, (1,)*3, 1)(x)
        x = x.reshape(-1)
        
        return x
    

class InceptBlock3D(nn.Module):
    """Inception block submodule"""
    filters: Sequence[int]
    filters_reduce: Sequence[int]
    strides: Union[None, int, Sequence[int]]
    dims: int
    do_5x5: bool = True
    do_3x3: bool = True
    #input_shape: Sequence[int]

    @nn.compact
    def __call__(self, x):

        f_red = self.filters_reduce

        outs = []
        
        if self.do_5x5:
        # 5x5 filter
         x1 = nn.Conv(features=f_red[0], kernel_size=(1,)*self.dims, strides=None)(x)
         #x1 = nn.Conv(features=self.filters[0], kernel_size=(1,5,5), strides=None)(x1)
         x1 = nn.Conv(features=self.filters[0], kernel_size=(3,5,5), strides=self.strides)(x1)
         outs.append(x1)
          
        if self.do_3x3:
        # 3x3 filter
          x2 = nn.Conv(features=f_red[1], kernel_size=(1,)*self.dims, strides=None)(x)
          x2 = nn.Conv(features=self.filters[1], kernel_size=(3,3,3), strides=self.strides)(x2)
          #x2 = nn.Conv(features=self.filters[1], kernel_size=(1,3,3), strides=self.strides)(x2)
          outs.append(x2)

        # 1x1
        x3 = nn.Conv(features=f_red[2], kernel_size=(1,)*self.dims, strides=None)(x)
        x3 = nn.Conv(features=self.filters[2], kernel_size=(1,)*self.dims, strides=self.strides)(x3)
        outs.append(x3)
        
        # maxpool and avgpool
        x4 = nn.max_pool(x, (3,)*self.dims, padding='SAME')
        x4 = nn.Conv(features=self.filters[3], kernel_size=(1,)*self.dims, strides=self.strides)(x4)
        outs.append(x4)
                    
        x = jnp.concatenate(outs, axis=-1)
        
        return x    
    

class ConvBlock3D(nn.Module):
    """Inception block submodule"""
    filters: int
    strides: int=2
    dims: int=3

    @nn.compact
    def __call__(self, x):

        fs = self.filters
        x = nn.Conv(features=fs, kernel_size=(3,)*self.dims, strides=None)(x)
        x = nn.Conv(features=fs, kernel_size=(3,)*self.dims, strides=self.strides)(x)
        
        return x    
    

class CNN3D(nn.Module):
    """An incept net architecture"""
    filters: Sequence[int] = (10,10,10,10) #(5,5,5,5) #(2,2,2,2)
    filters_reduce: Sequence[int] = (3, 3, 3, 3)
    div_factor: float = 0.005
    out_shape: int = 2
    do_big_convs: bool = True
    act: str = "almost_leaky"
    
    @nn.compact
    def __call__(self, x):
        fs = self.filters
        fs_red = self.filters_reduce
        dbg = self.do_big_convs
        if self.act == "almost_leaky":
           act = almost_leaky
        else:
           act = nn.elu

        x /= self.div_factor
        x = InceptBlock3D(fs, fs_red, strides=(1,2,2), dims=3, do_5x5=True)(x) # out: 4, 32, 32, 2
        x = act(x)
        #fs *= 2
        x = InceptBlock3D(fs, fs_red, strides=(1,2,2), dims=3, do_5x5=False)(x) # out: 4, 16, 16, 2
        x = act(x)
        #fs *= 4
        x = InceptBlock3D(fs, fs_red, strides=(1,2,2), dims=3, do_5x5=False)(x) # out: 4, 8, 8, 2
        x = act(x)
        #fs *= 2
        x = InceptBlock3D(fs, fs_red,  strides=(1,2,2), dims=3, do_5x5=False)(x) # out: 4, 4, 4, 2
        x = act(x)
        #fs *= 4
        x = InceptBlock3D(fs, fs_red,  strides=(2,2,2), dims=3, do_5x5=False)(x) # out: 1, 1, 1, 2
        x = act(x)
        #fs *= 2
        x = InceptBlock3D(fs, fs_red,  strides=(2,2,2), dims=3, do_5x5=False, do_3x3=False)(x) # out: 1, 1, 1, 2
        x = act(x)
        x = nn.Conv(self.out_shape, (1,)*3, 1)(x)
        x = x.reshape(-1)
        
        return x
    


class IncpetNet3DExplicit(nn.Module):
    """An incept net architecture"""
    filters: Sequence[int] = (8, 16, 16, 16)
    filters_reduce: Sequence[int] = (4, 8, 8, 8)
    div_factor: float = 0.005
    out_shape: int = 2
    do_big_convs: bool = True
    act: str = "almost_leaky"
    
    @nn.compact
    def __call__(self, x):
        fs = self.filters
        fs_red = self.filters_reduce
        dbg = self.do_big_convs
        if self.act == "almost_leaky":
           act = almost_leaky
        else:
           act = nn.elu

        x /= self.div_factor
        # fs = (5x5, 3x3, 1x1, maxpool)
        fs = self.filters
        fs_red = self.filters_reduce
        #x = InceptBlock3D((8,16,16,16), (4,8,8,8), strides=(1,2,2), dims=3, do_5x5=True)(x) # out: 4, 32, 32, 2
        x = ConvBlock3D(filters=32, strides=(1,2,2))(x) # 4, 16, 16
        x = act(x)
        x = ConvBlock3D(filters=32, strides=(1,2,2))(x) # 4, 16, 16
        x = act(x)
        x = ConvBlock3D(filters=32, strides=(1,2,2))(x) # 4, 8, 8
        x = act(x)
        x = ConvBlock3D(filters=32, strides=(1,2,2))(x) # 4, 4, 4
        x = act(x)
        x = ConvBlock3D(filters=32, strides=(2,2,2))(x) # 2, 2, 2
        x = act(x)
        x = ConvBlock3D(filters=32, strides=(2,2,2))(x) # 1, 1, 1
        x = act(x)
        x = nn.Conv(self.out_shape, (1,)*3, 1)(x)
        x = x.reshape(-1)
        
        return x