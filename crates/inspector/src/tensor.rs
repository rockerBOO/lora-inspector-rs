use candle_core::IndexOp;
use candle_core::{Result, Tensor};

// pub fn kron(a: &Tensor, b: &Tensor) -> Result<Tensor> {
//     let a_shape = a.shape();
//     let b_shape = b.shape();
//
//     // Get dimensions
//     let a_dims = a_shape.dims();
//     let b_dims = b_shape.dims();
//
//     // Handle different dimensionalities
//     match (a_dims.len(), b_dims.len()) {
//         (1, 1) => kron_1d(a, b),
//         (2, 2) => kron_2d(a, b),
//         (1, 2) => {
//             // Treat 1D tensor as row vector: [n] -> [1, n]
//             let a_2d = a.unsqueeze(0)?;
//             kron_2d(&a_2d, b)
//         }
//         (2, 1) => {
//             // Treat 1D tensor as column vector: [n] -> [n, 1]
//             let b_2d = b.unsqueeze(1)?;
//             kron_2d(a, &b_2d)
//         }
//         _ => kron_nd(a, b), // General N-dimensional case
//     }
// }

/// Kronecker product implementation for Candle
/// Equivalent to torch.kron(a, b)
pub fn kron(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let a_dims = a.shape().dims();
    let b_dims = b.shape().dims();

    match (a_dims.len(), b_dims.len()) {
        (1, 1) => kron_1d(a, b),
        (2, 2) => {
            // Use efficient version for larger tensors
            let total_elements = a.elem_count() * b.elem_count();
            if total_elements > 10000 {
                kron_2d_efficient(a, b)
            } else {
                kron_2d(a, b)
            }
        }
        (1, 2) => {
            let a_2d = a.unsqueeze(0)?;
            kron_2d_efficient(&a_2d, b)
        }
        (2, 1) => {
            let b_2d = b.unsqueeze(1)?;
            kron_2d_efficient(a, &b_2d)
        }
        _ => kron_nd(a, b),
    }
}

/// 1D Kronecker product
fn kron_1d(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let a_len = a.dim(0)?;
    let b_len = b.dim(0)?;

    // Reshape a to [a_len, 1] and b to [1, b_len] for broadcasting
    let a_reshaped = a.reshape(&[a_len, 1])?;
    let b_reshaped = b.reshape(&[1, b_len])?;

    // Element-wise multiplication with broadcasting: [a_len, 1] * [1, b_len] -> [a_len, b_len]
    let result_2d = a_reshaped.broadcast_mul(&b_reshaped)?;

    // Flatten to 1D: [a_len * b_len]
    result_2d.reshape(&[a_len * b_len])
}

/// 2D Kronecker product (most common case)
fn kron_2d(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let a_rows = a.dim(0)?;
    let a_cols = a.dim(1)?;
    let b_rows = b.dim(0)?;
    let b_cols = b.dim(1)?;

    let out_rows = a_rows * b_rows;
    let out_cols = a_cols * b_cols;

    // Method 1: Using explicit loops (clearer but potentially slower)
    let mut result_data = Vec::with_capacity(out_rows * out_cols);

    for i in 0..a_rows {
        for k in 0..b_rows {
            for j in 0..a_cols {
                for l in 0..b_cols {
                    let a_val = a.i((i, j))?.to_scalar::<f32>()?;
                    let b_val = b.i((k, l))?.to_scalar::<f32>()?;
                    result_data.push(a_val * b_val);
                }
            }
        }
    }

    Tensor::from_vec(result_data, &[out_rows, out_cols], a.device())
}

/// More efficient 2D implementation using tensor operations
fn kron_2d_efficient(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let a_rows = a.dim(0)?;
    let a_cols = a.dim(1)?;
    let b_rows = b.dim(0)?;
    let b_cols = b.dim(1)?;

    // Reshape a: [a_rows, a_cols] -> [a_rows, 1, a_cols, 1]
    let a_expanded = a.reshape(&[a_rows, 1, a_cols, 1])?;

    // Reshape b: [b_rows, b_cols] -> [1, b_rows, 1, b_cols]
    let b_expanded = b.reshape(&[1, b_rows, 1, b_cols])?;

    // Broadcast multiply: [a_rows, 1, a_cols, 1] * [1, b_rows, 1, b_cols]
    // -> [a_rows, b_rows, a_cols, b_cols]
    let result_4d = a_expanded.broadcast_mul(&b_expanded)?;

    // Reshape to final form: [a_rows * b_rows, a_cols * b_cols]
    result_4d.reshape(&[a_rows * b_rows, a_cols * b_cols])
}

/// General N-dimensional Kronecker product
fn kron_nd(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let a_dims = a.shape().dims();
    let b_dims = b.shape().dims();

    // Calculate output dimensions
    let mut out_dims = Vec::new();
    let max_len = a_dims.len().max(b_dims.len());

    for i in 0..max_len {
        let a_dim = if i < a_dims.len() { a_dims[i] } else { 1 };
        let b_dim = if i < b_dims.len() { b_dims[i] } else { 1 };
        out_dims.push(a_dim * b_dim);
    }

    // Pad dimensions to match
    let mut a_padded_dims = vec![1; max_len];
    let mut b_padded_dims = vec![1; max_len];

    for (i, &dim) in a_dims.iter().enumerate() {
        a_padded_dims[max_len - a_dims.len() + i] = dim;
    }
    for (i, &dim) in b_dims.iter().enumerate() {
        b_padded_dims[max_len - b_dims.len() + i] = dim;
    }

    // Reshape tensors for broadcasting
    let a_reshaped = a.reshape(a_padded_dims.clone())?;
    let b_reshaped = b.reshape(b_padded_dims.clone())?;

    // Create expanded dimensions for Kronecker product
    let mut a_expanded_dims = Vec::new();
    let mut b_expanded_dims = Vec::new();

    for i in 0..max_len {
        a_expanded_dims.push(a_padded_dims[i]);
        a_expanded_dims.push(1);
        b_expanded_dims.push(1);
        b_expanded_dims.push(b_padded_dims[i]);
    }

    let a_expanded = a_reshaped.reshape(a_expanded_dims)?;
    let b_expanded = b_reshaped.reshape(b_expanded_dims)?;

    // Broadcast multiply and reshape
    let result_expanded = a_expanded.broadcast_mul(&b_expanded)?;
    result_expanded.reshape(out_dims)
}
