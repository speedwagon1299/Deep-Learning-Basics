import numpy as np
np.random.seed(1)

def zero_pad(X, pad):
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=(0, 0))  
    # no pad on m, pad for n_H and n_W, no pad for n_c; pad_width for before and after dimension, square/cube autocompleted    
    return X_pad

def conv_single_step(a_slice_prev, W, b):
    s = a_slice_prev*W
    Z = np.sum(s)
    Z = Z + float(b)  # b is an array while Z is a scalar, so casting on b required
    return Z

def conv_forward(A_prev, W, b, hparameters):
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape
    stride = hparameters['stride']
    pad = hparameters['pad']
    
    n_H = int(((n_H_prev + 2*pad - f)/stride))+1
    n_W = int(((n_W_prev + 2*pad - f)/stride))+1
    Z = np.zeros((m,n_H,n_W,n_C))
    A_prev_pad = zero_pad(A_prev, pad)
    
    for i in range(m):
        for j in range(n_H):
            vert_start = stride*j
            vert_end = vert_start + f
            for k in range(n_W):
                horiz_start = stride*k
                horiz_end = horiz_start + f
                for c in range(n_C):
                    a_slice_prev = A_prev_pad[i,vert_start:vert_end,horiz_start:horiz_end,:]
                    weights = W[:,:,:,c]
                    bias = b[:,:,:,c]
                    Z[i,j,k,c] = conv_single_step(a_slice_prev, weights, bias)                
    cache = (A_prev, W, b, hparameters)
    return Z, cache

def pool_forward(A_prev, hparameters, mode = "max"):
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    f = hparameters["f"]
    stride = hparameters["stride"]
    
    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev
    A = np.zeros((m, n_H, n_W, n_C))              
 
    for i in range(m):
        for j in range(n_H):
            vert_start = j*stride
            vert_end = vert_start + f
            for k in range(n_W):
                horiz_start = k*stride
                horiz_end = horiz_start + f
                for c in range(n_C):
                    a_prev_slice = A_prev[i,vert_start:vert_end,horiz_start:horiz_end,c]
                    if mode == 'max':
                        A[i, j, k, c] = np.max(a_prev_slice)
                    if mode == 'average':
                        A[i, j, k, c] = np.mean(a_prev_slice)
    cache = (A_prev, hparameters)
    return A, cache
