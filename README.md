A library to compute N-D convolutions, transposed convolutions, recursive convolution in pytorch, using `Linear` filter, arbitrary `nn.Module` filter. It also gives the option of automaticly finding convolution parameters to match a desired output shape.

# Instalation

Use `pip3 install torchConvNd`

# Documentation

## convNd
```python
convNd(x, weight, kernel, stride=1, dilation=1, padding=0, bias=None, padding_mode='constant', padding_value=0)
```

N-Dimensional convolution.

#### *Inputs* :

__x__ : `torch.tensor` of shape `(batch_size, *shape)`.

__Weight__ : `torch.tensor` of size `(dilation[0]*dilation[1]*...dilation[n_dims], kernel[0]*kernel[1]*...kernel[n_dims])`.

__kernel__ : array-like or int, kernel size of the convolution.

__stride__ : array-like or int, stride length of the convolution.

__dilation__ : array-like or int, dilation of the convolution.

__padding__ : `None`, array-like or int, padding size.

__bias__ : `None` or `torch.tensor` of size `(dilation[0]*dilation[1]*...dilation[n_dims])`.

__padding\_mode__,  __padding\_value__: see [`pad`](#pad).

#### _Outputs_ :

__out__ : `torch.tensor` of shape `(batch_size, *shape_out)`.

## ConvNd
```python
ConvNd(kernel, stride=1, dilation=1, padding=0, bias=False, padding_mode='constant', padding_value=0)
```

Equivalent of [`convNd`](#convNd) as a `torch.nn.Module` class.

#### *Inputs* :

__bias__ : boolean, controls the usage or not of biases.

__kernel__, __stride__, __dilation__, __padding__, __padding\_mode__,  __padding\_value__: Same as in [`convNd`](#convNd).

## convTransposeNd
```python
convTransposeNd(x, weight, kernel, stride=1, dilation=1, padding=0, bias=None, padding_mode='constant', padding_value=0)
```

Transposed convolution (using [`repeat_intereleave`](https://pytorch.org/docs/stable/torch.html#torch.repeat_interleave)).

#### *Inputs* :

__x__ : `torch.tensor` of shape `(batch_size, *shape)`.

__Weight__ : `torch.tensor` of size `(dilation[0]*dilation[1]*...dilation[n_dims], kernel[0]*kernel[1]*...kernel[n_dims])`.

__kernel__ : array-like or int, kernel size of the transposed convolution.

__stride__ : array-like or int, stride length of the transposed convolution.

__dilation__ : array-like or int, dilation of the convolution.

__padding__ : `None`, array-like or int, padding size.

__bias__ : `None` or `torch.tensor` of size `(dilation[0]*dilation[1]*...dilation[n_dims])`.

__padding\_mode__,  __padding\_value__: see [`pad`](#pad).

#### _Outputs_ :

__out__ : `torch.tensor` of shape `(batch_size, *shape_out)`.

## ConvTransposeNd
```python
ConvTransposeNd(kernel, stride=1, dilation=1, padding=0, bias=None, padding_mode='constant', padding_value=0)
```

Equivalent of [`convTransposeNd`](#convTransposeNd) as a `torch.nn.Module` class.

#### *Inputs* :

__bias__ : boolean, controls the usage or not of biases.

__kernel__, __stride__, __dilation__, __padding__, __padding\_mode__,  __padding\_value__: Same as in [`convTransposeNd`](#convTransposeNd).


## convNdFunc
```python
convNdFunc(x, func, kernel, stride=1, padding=0, stride_transpose=1, padding_mode='constant', padding_value=0, *args)
```

Equivalent of [`convNd`](#convNd) using an arbitrary filter `func`.

#### *Inputs* :

__x__ : `torch.tensor` of shape `(batch_size, *shape)`.

__func__ : function, taking a `torch.tensor` of shape `(batch_size, *kernel)` and outputs a `torch.tensor` of shape `(batch_size,)`.

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

## convNdAuto
```python
convNdAuto(x, weight, shape, kernel, bias=None, padding_mode='constant', padding_value=0, max_dilation=3)
```

Equivalent of [`convNdAutoFunc`](#convNdAutoFunc) using a linear filter.

#### *Inputs* :

__x__ : `torch.tensor` of shape `(batch_size, *shape)`.

__Weight__ : `torch.tensor` of size `(dilation[0]*dilation[1]*...dilation[n_dims], kernel[0]*kernel[1]*...kernel[n_dims])`.

__shape__ : array-like or int, target shape of the convolution.

__kernel__ : array-like or int, kernel size of the  convolution.

__bias__ : `None` or `torch.tensor` of size `(dilation[0]*dilation[1]*...dilation[n_dims])`.

__padding\_mode__,  __padding\_value__: see [`pad`](#pad).

__max\_dilation__ : array-like or int, maximum value of dialtion.

__*args__: additional arguments to pass to `func`.

#### _Outputs_ :

__out__ : `torch.tensor` of shape `(batch_size, *shape_out)`, `shape_out` is strictly bigger than `shape`.

## ConvNdAuto
```python
ConvNdAuto(shape, kernel, bias=None, padding_mode='constant', padding_value=0, max_dilation=3)
```

Equivalent of [`convNdAuto`](#convNdAuto) as a `torch.nn.Module` class.

#### *Inputs* :

__bias__ : boolean, controls the usage or not of biases.

__shape__, __kernel__, __padding\_mode__, __padding\_value__, __max\_dilation__ : Same as in [`convNdAuto`](#convNdAuto).

## convNdAutoFunc
```python
convNdAutoFunc(x, shape, func, kernel, padding_mode='constant', padding_value=0, max_dilation=3, *args)
```

Uses [`autoShape`](#autoShape) to match the output shape to `shape`.

#### *Inputs* :

__x__ : `torch.tensor` of shape `(batch_size, *shape)`.

__shape__ : array-like or int, target shape of the convolution.

__func__ : function, taking a `torch.tensor` of shape `(batch_size, *kernel)` and outputs a `torch.tensor` of shape `(batch_size,)`.

__kernel__ : array-like or int, kernel size of the  convolution.

__padding\_mode__,  __padding\_value__: see [`pad`](#pad).

__max\_dilation__ : array-like or int, maximum value of dialtion.

__*args__: additional arguments to pass to `func`.

#### _Outputs_ :

__out__ : `torch.tensor` of shape `(batch_size, *shape_out)`, `shape_out` is strictly bigger than `shape`.

__*(additional returns)__ : any additional returns of `func`.

## ConvNdAutoFunc
```python
ConvNdAutoFunc(shape, func, kernel, padding_mode='constant', padding_value=0, max_dilation=3)
```

Equivalent of [`convNdAutoFunc`](#convNdAutoFunc) as a `torch.nn.Module` class.

__shape__, __func__, __kernel__, __padding\_mode__, __padding\_value__, __max\_dilation__ : Same as in [`convNdAutoFunc`](#convNdAutoFunc).

## convNdRec
```python
convNdRec(x, hidden, func, kernel, stride=1, dilation=1, padding=0, stride_transpose=1, padding_mode='constant', padding_value=0, *args):
```

Recursive version of [`convNdFunc`](#convNdFunc).

#### *Inputs* :

__x__ : `torch.tensor` of shape `(batch_size, seq, *shape)`.

__hidden__ : `torch.tensor` of shape `(length, *hidden_shape)` (if `length` < `batch_size` the tensor will be elongated with zeros).

__func__ : function, taking two `torch.tensor` of shape `(batch_size, seq, *kernel)` and `(batch_size, *hidden_shape)` and outputs two `torch.tensor` of shape `(batch_size, seq)` and `(batch_size, *hidden_shape)`.

__kernel__ : array-like or int, kernel size of the  convolution.

__stride__ : array-like or int, stride length of the convolution.

__dilation__ : array-like or int, dilation of the convolution.

__padding__ : `None`, array-like or int, padding size.

__stride\_transpose__ : array-like or int, equivalent to `stride` in [`convTransposeNd`](#convTransposeNd).

__padding\_mode__,  __padding\_value__: see [`pad`](#pad).

__*args__: additional arguments to pass to `func`.

#### _Outputs_ :

__out__ : `torch.tensor` of shape `(batch_size, seq, *shape_out)`.

__hidden__ : `torch.tensor` of shape `(batch_size, *hidden_shape)`.

__*(additional returns)__ : any additional returns of `func`.

## ConvNdRec
```python
ConvNdRec(x, hidden, func, kernel, stride=1, dilation=1, padding=0, stride_transpose=1, padding_mode='constant', padding_value=0, *args):
```

Equivalent of [`convNdRec`](#convNdRec) as a `torch.nn.Module` class.

#### *Inputs* :

__func__, __kernel__, __stride__, __dilation__, __padding__, __stride\_transpose__, __padding\_mode__, __padding\_value__ : Same as in [`convNdRec`](#convNdRec).

## convNdAutoRec
```python
convNdAutoRec(x, hidden, shape, func, kernel, padding_mode='constant', padding_value=0, max_dilation=3, *args)
```

Recursive version of [`convNdAutoFunc`](#convNdAutoFunc).

#### *Inputs* :

__x__ : `torch.tensor` of shape `(batch_size, seq, *shape)`.

__hidden__ : `torch.tensor` of shape `(length, *hidden_shape)` (if `length` < `batch_size` the tensor will be elongated with zeros).

__shape__ : array-like or int, target shape of the convolution.

__func__ : function, taking two `torch.tensor` of shape `(batch_size, seq, *kernel)` and `(batch_size, *hidden_shape)` and outputs two `torch.tensor` of shape `(batch_size, seq)` and `(batch_size, *hidden_shape)`.

__kernel__ : array-like or int, kernel size of the  convolution.

__padding\_mode__,  __padding\_value__: see [`pad`](#pad).

__max\_dilation__ : array-like or int, maximum value of dialtion.

__*args__: additional arguments to pass to `func`.

#### _Outputs_ :

__out__ : `torch.tensor` of shape `(batch_size, seq, *shape_out)`, `shape_out` is strictly bigger than `shape`.

__hidden__ : `torch.tensor` of shape `(batch_size, *hidden_shape)`.

__*(additional returns)__ : any additional returns of `func`.

## ConvNdAutoRec
```python
ConvNdAutoRec(shape, func, kernel, padding_mode='constant', padding_value=0, max_dilation=3)
```

Equivalent of [`convNdAutoRec`](#convNdAutoRec) as a `torch.nn.Module` class.

#### *Inputs* :

__shape__, __func__, __kernel__, __padding\_mode__, __padding\_value__, __max\_dilation__ : Same as in [`convNdAutoRec`](#convNdAutoRec).

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