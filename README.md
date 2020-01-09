A library to compute N-D convolutions and transposed convolutions in pytorch, using `Linear` filter, arbitrary `nn.Module` filter.

# Instalation

Use `pip3 install torchConvNd`

# Documentation

### convNd
```python
convNd(input, weight, kernel, stride=1, dilation=1, padding=0, bias=None, padding_mode='constant', padding_value=0)
```

__Weight :__ `torch.tensor` of size `(dilation[0]*dilation[1]*...dilation[n_dims], kernel[0]*kernel[1]*...kernel[n_dims])`.

__kernel__ : array-like or int, kernel size of the  convolution.

__stride__ : array-like or int, stride length of the convolution.

__dilation__ : array-like or int, output shape of the convolution.

__padding__ : `None`, array-like or int, padding size. If `None` the padding will be calculated to conserve the shape of the inpute tensor (assuming dilation and stride are identical).

__bias__ : `None` or `torch.tensor` of size `(dilation[0]*dilation[1]*...dilation[n_dims])`.

__padding\_mode__,  __padding\_value__: see [`pad`](#pad).

### ConvNd
```python
ConvNd(kernel, stride=1, dilation=1, padding=0, bias=False, padding_mode='constant', padding_value=0)
```

Equivalent of [`convNd`](#convNd) as a `torch.nn.Module` class.

###convTransposeNd
```python
convTransposeNd(x, weight, kernel, stride=1, dilation=1, padding=0, bias=None, padding_mode='constant', padding_value=0)
```

Transposed convolution (using [`repeat_intereleave`](https://pytorch.org/docs/stable/torch.html#torch.repeat_interleave)).

###ConvTransposeNd
```python
ConvTransposeNd(kernel, stride=1, dilation=1, padding=0, bias=None, padding_mode='constant', padding_value=0)
```

Equivalent of [`convTransposeNd`](#convTransposeNd) as a `torch.nn.Module` class.

### convNdFunc
```python
convNdFunc(input, func, kernel, stride=1, padding=0, padding_mode='constant', padding_value=0, *args)
```

__kernel__, __stride__, __padding__: see [`convNd`](#convNd).

__padding\_mode__,  __padding\_value__: see [`pad`](#pad).

### ConvNdFunc
```python
ConvNdFunc(func, kernel, stride=1, padding=0, padding_mode='constant', padding_value=0)
```

Equivalent of [`ConvNdFunc`](#ConvNdFunc) as a `torch.nn.Module` class.

###convTransposeNdFunc
```python
convNdFunc(x, func, kernel, stride=1, dilation=1, padding=0, padding_mode='constant', padding_value=0, *args)
```

Transposed convolution (using [`repeat_intereleave`](https://pytorch.org/docs/stable/torch.html#torch.repeat_interleave)).

###ConvTransposeNdFunc
```python
ConvNdFunc(func, kernel, stride=1, dilation=1, padding=0, padding_mode='constant', padding_value=0)
```

Equivalent of [`convTransposeNdFunc`](#convTransposeNdFunc) as a `torch.nn.Module` class.

# torchConvNd.Utils

### listify
```python
listify(x, dims=1)
```

Transform `x` to an iterable if it is not.

### pad
```python
pad(input, padding, mode='constant', value=0)
```

Equivalent to [torch.nn.functional.pad](https://pytorch.org/docs/stable/nn.functional.html#pad).

### Pad
```python
Pad(padding, mode='constant', value=0)
```

Return the function `lambda input: pad(input, padding, mode, value)`.

### view
```python
view(input, kernel, stride=1)
```

Generate a view (for a convolution) with parameters `kernel` and `stride`.

__kernel__, __stride__ : see [`convNd`](#convNd).

### View
```python
View(kernel, stride=1)
```

Return the function `lambda input: view(input, kernel, stride)`.

### Flatten
```python
Flatten()
```

A `torch.nn.Module` class that takes a tensor of shape `(N, i, j, k...)` and reshape it to `(N, i*j*k*...)`.

### Reshape
```python
Reshape(shape)
```

A `torch.nn.Module` class that takes a tensor of shape `(N, i)` and reshape it to `(N, shape[0], shape[1], ...)`.
