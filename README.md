A library to compute N-D convolutions and transposed convolutions in pytorch, using `Linear` filter, arbitrary `nn.Module` filter.

# Instalation

Use `pip3 install torchConvNd`

# Documentation

## convNd
```python
convNd(x, weight, kernel, stride=1, dilation=1, padding=0, bias=None, padding_mode='constant', padding_value=0)
```

__x__ : `torch.tensor` of shape `(batch_size, *shape)`.

__Weight__ : `torch.tensor` of size `(dilation[0]*dilation[1]*...dilation[n_dims], kernel[0]*kernel[1]*...kernel[n_dims])`.

__kernel__ : array-like or int, kernel size of the convolution.

__stride__ : array-like or int, stride length of the convolution.

__dilation__ : array-like or int, dilation of the convolution.

__padding__ : `None`, array-like or int, padding size.

__bias__ : `None` or `torch.tensor` of size `(dilation[0]*dilation[1]*...dilation[n_dims])`.

__padding\_mode__,  __padding\_value__: see [`pad`](#pad).

## ConvNd
```python
ConvNd(kernel, stride=1, dilation=1, padding=0, bias=False, padding_mode='constant', padding_value=0)
```

Equivalent of [`convNd`](#convNd) as a `torch.nn.Module` class.

__bias__ : boolean, controls the usage or not of biases.

__kernel__, __stride__, __dilation__, __padding__, __padding\_mode__,  __padding\_value__: Same as in [`convNd`](#convNd).

## convTransposeNd
```python
convTransposeNd(x, weight, kernel, stride=1, dilation=1, padding=0, bias=None, padding_mode='constant', padding_value=0)
```

Transposed convolution (using [`repeat_intereleave`](https://pytorch.org/docs/stable/torch.html#torch.repeat_interleave)).

__x__ : `torch.tensor` of shape `(batch_size, *shape)`.

__Weight__ : `torch.tensor` of size `(dilation[0]*dilation[1]*...dilation[n_dims], kernel[0]*kernel[1]*...kernel[n_dims])`.

__kernel__ : array-like or int, kernel size of the transposed convolution.

__stride__ : array-like or int, stride length of the transposed convolution.

__dilation__ : array-like or int, dilation of the convolution.

__padding__ : `None`, array-like or int, padding size.

__bias__ : `None` or `torch.tensor` of size `(dilation[0]*dilation[1]*...dilation[n_dims])`.

__padding\_mode__,  __padding\_value__: see [`pad`](#pad).

## ConvTransposeNd
```python
ConvTransposeNd(kernel, stride=1, dilation=1, padding=0, bias=None, padding_mode='constant', padding_value=0)
```

Equivalent of [`convTransposeNd`](#convTransposeNd) as a `torch.nn.Module` class.

__bias__ : boolean, controls the usage or not of biases.

__kernel__, __stride__, __dilation__, __padding__, __padding\_mode__,  __padding\_value__: Same as in [`convTransposeNd`](#convTransposeNd).


## convNdFunc
```python
convNdFunc(x, func, kernel, stride=1, padding=0, stride_transpose=1, padding_mode='constant', padding_value=0, *args)
```

__x__ : `torch.tensor` of shape `(batch_size, *shape)`.

__func__ : function, taking a `torch.tensor` of shape `(batch_size, *kernel)` and outputs a `torch.tensor` of shape `(batch_size,)`.

__kernel__ : array-like or int, kernel size of the  convolution.

__stride__ : array-like or int, stride length of the convolution.

__dilation__ : array-like or int, dilation of the convolution.

__padding__ : `None`, array-like or int, padding size.

__stride\_transpose__ : array-like or int, equivalent to `stride` in [`convTransposeNd`](#convTransposeNd).

__padding\_mode__,  __padding\_value__: see [`pad`](#pad).

__*args__: additional arguments to pass to `func`.

## ConvNdFunc
```python
ConvNdFunc(func, kernel, stride=1, padding=0, padding_mode='constant', padding_value=0)
```

Equivalent of [`ConvNdFunc`](#ConvNdFunc) as a `torch.nn.Module` class.

__func__, __kernel__, __stride__, __dilation__, __padding__, __stride\_transpose__, __padding\_mode__, __padding\_value__ : Same as in [`convNdFunc`](#convNdFunc).

## convNdAutoFunc
```python
convNdAutoFunc(x, shape, func, kernel, padding_mode='constant', padding_value=0, max_dilation=3, max_stride_transpose=4, Clip=False, *args)
```

Uses [`autoShape`](#autoShape) to match `shape` at each convolution, with `shape` a target shape inside `shapes`.

__x__ : `torch.tensor` of shape `(batch_size, *shape)`.

__shape__ : array-like or int, target shape of the convolution.

__func__ : function, taking a `torch.tensor` of shape `(batch_size, *kernel)` and outputs a `torch.tensor` of shape `(batch_size,)`.

__kernel__ : array-like or int, kernel size of the  convolution.

__padding\_mode__,  __padding\_value__: see [`pad`](#pad).

__max\_dilation__ : array-like or int, maximum value of dialtion.

__max\_stride\_transpose__ : array-like or int, maximum value of stride_transpose.

__Clip__ : boolean, if true clips the output to exactly match `shape`.

__*args__: additional arguments to pass to `func`.

## ConvNdAutoFunc
```python
ConvNdAutoFunc(func, kernel, padding_mode='constant', padding_value=0, max_dilation=3, max_stride_transpose=4, Clip=False)
```

Equivalent of [`convNdAutoFunc`](#convNdAutoFunc) as a `torch.nn.Module` class.

__func__, __kernel__, __padding\_mode__, __padding\_value__, __max\_dilation__, __max\_stride\_transpose__, __Clip__ : Same as in [`convNdAutoFunc`](#convNdAutoFunc).


## convNdAuto
```python
convNdAuto(x, weight, shapes, kernel, bias=None, padding_mode='constant', padding_value=0, max_dilation=3, max_stride_transpose=4, Clip=False)
```

Equivalent of [`convNdAutoFunc`](#convNdAutoFunc) using a linear filter.

__x__ : `torch.tensor` of shape `(batch_size, *shape)`.

__Weight__ : `torch.tensor` of size `(dilation[0]*dilation[1]*...dilation[n_dims], kernel[0]*kernel[1]*...kernel[n_dims])`.

__shape__ : array-like or int, target shape of the convolution.

__kernel__ : array-like or int, kernel size of the  convolution.

__bias__ : `None` or `torch.tensor` of size `(dilation[0]*dilation[1]*...dilation[n_dims])`.

__padding\_mode__,  __padding\_value__: see [`pad`](#pad).

__max\_dilation__ : array-like or int, maximum value of dialtion.

__max\_stride\_transpose__ : array-like or int, maximum value of stride_transpose.

__Clip__ : boolean, if true clips the output to exactly match `shape`.

__*args__: additional arguments to pass to `func`.

## ConvNdAuto
```python
ConvNdAuto(kernel, bias=None, padding_mode='constant', padding_value=0, max_dilation=3, max_stride_transpose=4, Clip=False)
```

Equivalent of [`convNdAuto`](#convNdAuto) as a `torch.nn.Module` class.

__bias__ : boolean, controls the usage or not of biases.

__kernel__, __padding\_mode__, __padding\_value__, __max\_dilation__, __max\_stride\_transpose__, __Clip__ : Same as in [`convNdAuto`](#convNdAuto).

## convNdRec
```python
convNdRec(x, hidden, func, kernel, stride=1, dilation=1, padding=0, stride_transpose=1, padding_mode='constant', padding_value=0, *args):
```

Recursive version of [`convNdFunc`](#convNdFunc).

__x__ : `torch.tensor` of shape `(batch_size, *shape)`.

__hidden__ : `torch.tensor` of shape `(length, *hidden_shape)` (if `length` < `batch_size` the tensor will be elongated with zeros).

__func__ : function, taking two `torch.tensor` of shape `(batch_size, *kernel)` and `(batch_size, *hidden_shape)` and outputs two `torch.tensor` of shape `(batch_size,)` and `(batch_size, *hidden_shape)`.

__kernel__ : array-like or int, kernel size of the  convolution.

__stride__ : array-like or int, stride length of the convolution.

__dilation__ : array-like or int, dilation of the convolution.

__padding__ : `None`, array-like or int, padding size.

__stride\_transpose__ : array-like or int, equivalent to `stride` in [`convTransposeNd`](#convTransposeNd).

__padding\_mode__,  __padding\_value__: see [`pad`](#pad).

__*args__: additional arguments to pass to `func`.

## ConvNdRec
```python
ConvNdRec(x, hidden, func, kernel, stride=1, dilation=1, padding=0, stride_transpose=1, padding_mode='constant', padding_value=0, *args):
```

Equivalent of [`convNdRec`](#convNdRec) as a `torch.nn.Module` class.

__func__, __kernel__, __stride__, __dilation__, __padding__, __stride\_transpose__, __padding\_mode__, __padding\_value__ : Same as in [`convNdRec`](#convNdRec).

## convNdAutoRec
```python
convNdAutoRec(x, hidden, shape, func, kernel, padding_mode='constant', padding_value=0, max_dilation=3, max_stride_transpose=4, *args)
```

Recursive version of [`convNdAutoFunc`](#convNdAutoFunc).

__x__ : `torch.tensor` of shape `(batch_size, *shape)`.

__hidden__ : `torch.tensor` of shape `(length, *hidden_shape)` (if `length` < `batch_size` the tensor will be elongated with zeros).

__shape__ : array-like or int, target shape of the convolution.

__func__ : function, taking two `torch.tensor` of shape `(batch_size, *kernel)` and `(batch_size, *hidden_shape)` and outputs two `torch.tensor` of shape `(batch_size,)` and `(batch_size, *hidden_shape)`.

__kernel__ : array-like or int, kernel size of the  convolution.

__padding\_mode__,  __padding\_value__: see [`pad`](#pad).

__max\_dilation__ : array-like or int, maximum value of dialtion.

__max\_stride\_transpose__ : array-like or int, maximum value of stride_transpose.

__Clip__ : boolean, if true clips the output to exactly match `shape`.

__*args__: additional arguments to pass to `func`.

## ConvNdAutoRec
```python
ConvNdAutoRec(func, kernel, padding_mode='constant', padding_value=0, max_dilation=3, max_stride_transpose=4)
```

Equivalent of [`convNdAutoRec`](#convNdAutoRec) as a `torch.nn.Module` class.

__func__, __kernel__, __padding\_mode__, __padding\_value__, __max\_dilation__, __max\_stride\_transpose__, __Clip__ : Same as in [`convNdAutoRec`](#convNdAutoRec).

# torchConvNd.Utils

## listify
```python
listify(x, dims=1)
```

Transform `x` to an iterable if it is not.

__x__ : array like or non iterable object (or string), object to listify.

__dims__ : int, array size to obtain.

## convShape
```python
convShape(input_shape, kernel, stride=1, dilation=1, padding=0, stride_transpose=1)
```

Compute the ouput shape of a convolution.

__input\_shape__ : array-like or int, shape of the input tensor.

__kernel__ : array-like or int, kernel size of the convolution.

__stride__ : array-like or int, stride length of the convolution.

__dilation__ : array-like or int, dilation of the convolution.

__padding__ : `None`, array-like or int, padding size.

__stride\_transpose__ : array-like or int, equivalent to `stride` in [`convTransposeNd`](#convTransposeNd).

## autoShape
```python
autoShape(input_shape, kernel, output_shape, max_dilation=3, max_stride_transpose=4)
```

Compute the optimal parameters `stride`, `dilation`, `padding` and `stride_transpose` to match `output_shape`.

__input\_shape__ : array-like or int, shape of the input tensor.

__kernel__ : array-like or int, kernel size of the  convolution.

__output\_shape__ : array-like or int, target shape of the convolution.

__max\_dilation__ : array-like or int, maximum value of dialtion.

__max\_stride\_transpose__ : array-like or int, maximum value of stride_transpose.

## clip
```python
clip(x, shape)
```

Take a slice of `x` of size `shape` (in the center).

__x__ :  `torch.tensor`, tensor to clip.

__shape__ : array-like or int, shape to obtain.

## Clip
```python
Clip(shape)
```

Equivalent of [`clip`](#clip) which returns a function.

__shape__ : same as in [`clip`](#clip).

## pad
```python
pad(x, padding, padding_mode='constant', padding_value=0)
```

Based on [torch.nn.functional.pad](https://pytorch.org/docs/stable/nn.functional.html#pad).

__x__ :  `torch.tensor`, tensor to clip.

__padding__ : array-like or int, size of the padding (identical on each size).

__padding\_mode__ : 'constant', 'reflect', 'replicate' or 'circular', see [torch.nn.functional.pad](https://pytorch.org/docs/stable/nn.functional.html#pad).

__padding\_value__ : float, value to pad with if `padding_mode` id 'constant'.

## Pad
```python
Pad(padding, padding_mode='constant', padding_value=0)
```

Equivalent of [`pad`](#pad) which returns a function.

__padding__, __padding\_mode__, __padding\_value__ : same as with [`pad`](#pad)

## view
```python
view(x, kernel, stride=1)
```

Generate a view (for a convolution) with parameters `kernel` and `stride`.

__x__ :  `torch.tensor`, tensor to clip.

__kernel__ : array-like or int, kernel size of the convolution.

__stride__ : array-like or int, stride length of the convolution.

## View
```python
View(kernel, stride=1)
```

Equivalent of [`view`](#view) which returns a function.

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

__shape__ : array-like or int, shape to obtain.