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

__dilation__ : array-like or int, dilation of the convolution.

__padding__ : `None`, array-like or int, padding size.

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

__kernel__, __stride__, __dilation__, __padding__: see [`convNd`](#convNd).

__padding\_mode__,  __padding\_value__: see [`pad`](#pad).

__stride\_transpose__ : equivalent of `stride` for [`convTransposeNd`](#convTransposeNd).

### ConvNdFunc
```python
ConvNdFunc(func, kernel, stride=1, padding=0, padding_mode='constant', padding_value=0)
```

Equivalent of [`ConvNdFunc`](#ConvNdFunc) as a `torch.nn.Module` class.

### convNdAutoFunc
```python
convNdAutoFunc(x, shapes, func, kernel, padding_mode='constant', padding_value=0, max_dilation=3, max_stride_transpose=4, *args)
```

Uses [`autoShape`](#autoShape) to match `shape` at each convolution, with `shape` a target shape inside `shapes`.

### ConvNdAutoFunc
```python
ConvNdAutoFunc(func, kernel, padding_mode='constant', padding_value=0, max_dilation=3, max_stride_transpose=4)
```

Equivalent of [`convNdAutoFunc`](#convNdAutoFunc) as a `torch.nn.Module` class.

### convNdAuto
```python
convNdAuto(x, weight, shapes, kernel, bias=None, padding_mode='constant', padding_value=0, max_dilation=3, max_stride_transpose=4)
```

Equivalent of [`convNdAutoFunc`](#convNdAutoFunc) using a linear filter.

### ConvNdAuto
```python
convNdAuto(kernel, bias=None, padding_mode='constant', padding_value=0, max_dilation=3, max_stride_transpose=4)
```

Equivalent of [`convNdAuto`](#convNdAuto) as a `torch.nn.Module` class.

### convNdRec
```python
convNdRec(x, hidden, shapes, func, kernel, padding_mode='constant', padding_value=0, max_dilation=3, max_stride_transpose=4, *args)
```

Recursive version of [`convNdAutoFunc`](#convNdAutoFunc).

### ConvNdRec
```python
ConvNdRec(func, kernel, padding_mode='constant', padding_value=0, max_dilation=3, max_stride_transpose=4)
```

Equivalent of [`convNdRec`](#convNdRec) as a `torch.nn.Module` class.

# torchConvNd.Utils

### listify
```python
listify(x, dims=1)
```

Transform `x` to an iterable if it is not.

### convShape
```python
convShape(input_shape, kernel, stride=1, dilation=1, padding=0, stride_transpose=1)
```

Compute the ouput shape of a convolution of parameters `kernel`, `stride`, `dilation`, `padding` and `stride_transpose` given an input of shape `input_shape`.

### autoShape
```python
autoShape(input_shape, kernel, output_shape, max_dilation=3, max_stride_transpose=4)
```

Compute the optimal parameters `stride`, `dilation`, `padding` and `stride_transpose` given `input_shape`, `kernel` to match `output_shape`.

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
