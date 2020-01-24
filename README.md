A library to compute N-D convolutions, transposed convolutions and recursive convolution in pytorch, using `Linear` filter or arbitrary functions as filter. It also gives the option of automaticly finding convolution parameters to match a desired output shape.

# Instalation

Use `pip3 install torchConvNd`

# Documentation

## convNd
```python
convNd(x, weight, kernel, stride=1, dilation=1, padding=0, bias=None, padding_mode='constant', padding_value=0)
```

N-Dimensional convolution.

#### *Inputs* :

__x__ : `torch.tensor` of shape `(batch_size, C_in, *shape)`.

__Weight__ : `torch.tensor` of size `(C_in * kernel[0] * kernel[1] * ...kernel[n_dims], C_out)`.

__kernel__ : array-like or int, kernel size of the convolution.

__stride__ : array-like or int, stride length of the convolution.

__dilation__ : array-like or int, dilation of the convolution.

__padding__ : `None`, array-like or int, padding size.

__bias__ : `None` or `torch.tensor` of size `(C_out, )`.

__padding\_mode__,  __padding\_value__: see [`pad`](#pad).

#### _Outputs_ :

__out__ : `torch.tensor` of shape `(batch_size, C_out, *shape_out)`.

## ConvNd
```python
ConvNd(in_channels, out_channels, kernel, stride=1, dilation=1, padding=0, bias=False, padding_mode='constant', padding_value=0)
```

Equivalent of [`convNd`](#convNd) as a `torch.nn.Module` class.

#### *Inputs* :

__in\_channels__ : int, number of in channels.

__out\_channels__ : int, number of out channels.

__bias__ : boolean, controls the usage or not of biases.

__kernel__, __stride__, __dilation__, __padding__, __padding\_mode__,  __padding\_value__: Same as in [`convNd`](#convNd).

## convTransposeNd
```python
convTransposeNd(x, weight, kernel, stride=1, dilation=1, padding=0, bias=None, padding_mode='constant', padding_value=0)
```

Transposed convolution (using [`repeat_intereleave`](https://pytorch.org/docs/stable/torch.html#torch.repeat_interleave)).

#### *Inputs* :

__x__ : `torch.tensor` of shape `(batch_size, C_in, *shape)`.

__Weight__ : `torch.tensor` of size `(C_in * kernel[0] * kernel[1] * ...kernel[n_dims], C_out)`.

__kernel__ : array-like or int, kernel size of the transposed convolution.

__stride__ : array-like or int, stride length of the transposed convolution.

__dilation__ : array-like or int, dilation of the convolution.

__padding__ : `None`, array-like or int, padding size.

__bias__ : `None` or `torch.tensor` of size `(C_out, )`.

__padding\_mode__,  __padding\_value__: see [`pad`](#pad).

#### _Outputs_ :

__out__ : `torch.tensor` of shape `(batch_size, *shape_out)`.

## ConvTransposeNd
```python
ConvTransposeNd(in_channels, out_channels, kernel, stride=1, dilation=1, padding=0, bias=None, padding_mode='constant', padding_value=0)
```

Equivalent of [`convTransposeNd`](#convTransposeNd) as a `torch.nn.Module` class.

#### *Inputs* :

__in\_channels__ : int, number of in channels.

__out\_channels__ : int, number of out channels.

__bias__ : boolean, controls the usage or not of biases.

__kernel__, __stride__, __dilation__, __padding__, __padding\_mode__,  __padding\_value__: Same as in [`convTransposeNd`](#convTransposeNd).


## convNdFunc
```python
convNdFunc(x, func, kernel, stride=1, padding=0, stride_transpose=1, padding_mode='constant', padding_value=0, *args)
```

Equivalent of [`convNd`](#convNd) using an arbitrary filter `func`.

#### *Inputs* :

__x__ : `torch.tensor` of shape `(batch_size, C_in, *shape)`.

__func__ : function, taking a `torch.tensor` of shape `(batch_size, C_in, *kernel)` and outputs a `torch.tensor` of shape `(batch_size, C_out)`.

__kernel__ : array-like or int, kernel size of the  convolution.

__stride__ : array-like or int, stride length of the convolution.

__dilation__ : array-like or int, dilation of the convolution.

__padding__ : `None`, array-like or int, padding size.

__stride\_transpose__ : array-like or int, equivalent to `stride` in [`convTransposeNd`](#convTransposeNd).

__padding\_mode__,  __padding\_value__: see [`pad`](#pad).

__*args__: additional arguments to pass to `func`.

#### _Outputs_ :

__out__ : `torch.tensor` of shape `(batch_size, *shape_out)`.

__*(additional returns)__ : any additional returns of `func`.

## ConvNdFunc
```python
ConvNdFunc(func, kernel, stride=1, padding=0, padding_mode='constant', padding_value=0)
```

Equivalent of [`convNdFunc`](#convNdFunc) as a `torch.nn.Module` class.

#### *Inputs* :

__func__, __kernel__, __stride__, __dilation__, __padding__, __stride\_transpose__, __padding\_mode__, __padding\_value__ : Same as in [`convNdFunc`](#convNdFunc).

# torchConvNd.Utils

## listify
```python
listify(x, dims=1)
```

Transform `x` to an iterable if it is not.

#### *Inputs* :

__x__ : array like or non iterable object (or string), object to listify.

__dims__ : int, array size to obtain.

#### _Outputs_ :

__out__ :  array like, listified version of x.

## convShape
```python
convShape(input_shape, kernel, stride=1, dilation=1, padding=0, stride_transpose=1)
```

Compute the ouput shape of a convolution.

#### *Inputs* :

__input\_shape__ : array-like or int, shape of the input tensor.

__kernel__ : array-like or int, kernel size of the convolution.

__stride__ : array-like or int, stride length of the convolution.

__dilation__ : array-like or int, dilation of the convolution.

__padding__ : `None`, array-like or int, padding size.

__stride\_transpose__ : array-like or int, equivalent to `stride` in [`convTransposeNd`](#convTransposeNd).

#### _Outputs_ :

__shape__ : array-like or int, predicted output shape of the convolution.

## autoShape
```python
autoShape(input_shape, kernel, output_shape, max_dilation=3)
```

Compute the optimal parameters `stride`, `dilation`, `padding` and `stride_transpose` to match `output_shape`.

#### *Inputs* :

__input\_shape__ : array-like or int, shape of the input tensor.

__kernel__ : array-like or int, kernel size of the  convolution.

__output\_shape__ : array-like or int, target shape of the convolution.

__max\_dilation__ : array-like or int, maximum value of dialtion.

#### _Outputs_ :

__kernel__ : array-like or int, `listified(kernel, len(input_shape))` if `input_shape` is a list, else `kernel`.

__stride__ : array-like or int, stride length of the convolution.

__dilation__ : array-like or int, dilation of the convolution.

__padding__ : array-like or int, padding size.

__stride\_transpose__ : array-like or int, equivalent to `stride` in [`convTransposeNd`](#convTransposeNd).

## pad
```python
pad(x, padding, padding_mode='constant', padding_value=0)
```

Based on [torch.nn.functional.pad](https://pytorch.org/docs/stable/nn.functional.html#pad).

#### *Inputs* :

__x__ :  `torch.tensor`, input tensor.

__padding__ : array-like or int, size of the padding (identical on each size).

__padding\_mode__ : 'constant', 'reflect', 'replicate' or 'circular', see [torch.nn.functional.pad](https://pytorch.org/docs/stable/nn.functional.html#pad).

__padding\_value__ : float, value to pad with if `padding_mode` id 'constant'.

#### _Outputs_ :

__out__ :  `torch.tensor`, paded tensor.

## Pad
```python
Pad(padding, padding_mode='constant', padding_value=0)
```

Equivalent of [`pad`](#pad) which returns a function.

#### *Inputs* :

__padding__, __padding\_mode__, __padding\_value__ : same as with [`pad`](#pad)

## view
```python
view(x, kernel, stride=1)
```

Generate a view (for a convolution) with parameters `kernel` and `stride`.

#### *Inputs* :

__x__ :  `torch.tensor`, input tensor.

__kernel__ : array-like or int, kernel size of the convolution.

__stride__ : array-like or int, stride length of the convolution.

#### *Outputs* :

__out__ :  `torch.tensor`, strided tensor.

## View
```python
View(kernel, stride=1)
```

Equivalent of [`view`](#view) which returns a function.

#### *Inputs* :

__kernel__, __stride__ : same as in [`view`](#view).

## Flatten
```python
Flatten()
```

A `torch.nn.Module` class that takes a tensor of shape `(N, i, j, k...)` and reshape it to `(N, i*j*k*...)`.

## Reshape
```python
Reshape(shape)
```

A `torch.nn.Module` class that takes a tensor of shape `(N, i)` and reshape it to `(N, *shape)`.

#### *Inputs* :

__shape__ : array-like or int, shape to obtain.

## Clip
```python
Clip(shape)
```

A `torch.nn.Module` that takes a slice of a tensor of size `shape` (in the center).

#### *Inputs* :

__shape__ : array-like or int, shape to obtain (doesn't affect an axis where `shape=-1`).