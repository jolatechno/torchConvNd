# torchConvNd

### convNd
```python
convNd(input, weight, kernel, stride=1, dilation=1, padding=0, bias=None, padding_mode='constant', padding_value=0)
```

__Weight :__ `torch.tensor` of size `(dilation[0]*dilation[1]*...dilation[n_dims], kernel[0]*kernel[1]*...kernel[n_dims])`.

__kernel__ : array-like or int, kernel size of the  convolution .

__stride__ : array-like or int, stride length of the convolution.

__padding__ : `None`, array-like or int, padding size. If `None` the padding will be calculated to conserve the shape of the inpute tensor (assuming dilation and stride are identical).

__bias__ : `None` or `torch.tensor` of size `(dilation[0]*dilation[1]*...dilation[n_dims])`.

__padding\_mode__ : see [torch.nn.functional.pad](https://pytorch.org/docs/stable/nn.functional.html#pad).

### ConvNd
```python
ConvNd(kernel, stride=1, dilation=1, padding=0, bias=False, padding_mode='constant', padding_value=0)
```

Equivalent of `convNd` as a `torch.nn.Module` class.

### convNdFunc
```python
convNdFunc(input, func, kernel, stride=1, padding=0, padding_mode='constant', padding_value=0, *args)
```

### ConvNdFunc
```python
ConvNdFunc(func, kernel, stride=1, padding=0, padding_mode='constant', padding_value=0)
```

### convNdRec
```python
convNdRec(input, mem, func, kernel, stride=1, padding=0, padding_mode='constant', padding_value=0, *args)
```

### ConvNdRec
```python
ConvNdRec(func, kernel,  stride=1, padding=0, padding_mode='constant', padding_value=0)
```

# torchConvNd.Utils

### listify
```python
listify(x, dims=1)
```

### sequencify
```python
sequencify(x, nlist=1, dims=1)
```

### extendedLen
```python
extendedLen(x)
```

### calcPadding
```python
calcPadding(kernel, stride)
```

### pad
```python
pad(input, padding, mode='constant', value=0)
```

### Pad
```python
Pad(padding, mode='constant', value=0)
```

### view
```python
view(input, kernel, stride=1)
```

### View
```python
View(kernel, stride=1)
```

### convPrep
```python
convPrep(input, kernel, stride=1, padding=0, padding_mode='constant', padding_value=0)
```

### ConvPrep
```python
ConvPrep(input, kernel, stride=1, padding=0, padding_mode='constant', padding_value=0)
```

### convPost(input, shape)
```python
convPost(input, shape)
```

### Flatten
```python
Flatten()
```

### Reshape
```python
Reshape(shape)
```