# torchConvNd

### convNd
```python
convNd(input, weight, kernel, stride=1, dilation=1, padding=0, bias=None, padding_mode='constant', padding_value=0)
```

### ConvNd
```python
ConvNd(kernel, stride=1, dilation=1, padding=0, bias=False, padding_mode='constant', padding_value=0)
```

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

## Utils

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