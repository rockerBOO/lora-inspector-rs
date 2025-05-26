use candle_core::Tensor;

#[cfg(test)]
/// torch.allclose() with custom tolerances
/// Formula: |a - b| <= atol + rtol * |b|
pub fn allclose_with_tol(
    a: &Tensor,
    b: &Tensor,
    rtol: f64,
    atol: f64,
) -> candle_core::Result<bool> {
    // Check shapes match
    if a.shape() != b.shape() {
        return Ok(false);
    }

    // Calculate absolute difference: |a - b|
    let diff = a.sub(b)?.abs()?;

    // Calculate tolerance: atol + rtol * |b|
    let b_abs = b.abs()?;
    let rtol_tensor = Tensor::new(rtol as f32, a.device())?.broadcast_as(b_abs.shape())?;
    let atol_tensor = Tensor::new(atol as f32, a.device())?.broadcast_as(b_abs.shape())?;
    let tolerance = atol_tensor.add(&b_abs.mul(&rtol_tensor)?)?;

    // Check if all elements satisfy: |a - b| <= tolerance
    let within_tolerance = diff.le(&tolerance)?;

    // Check if all values are true
    let all_true = within_tolerance.sum_all()?.to_scalar::<f32>()?;
    let total_elements = within_tolerance.elem_count() as f32;

    Ok((all_true - total_elements).abs() < 1e-6)
}

#[cfg(test)]
#[allow(dead_code)]
pub fn assert_allclose(
    a: &Tensor,
    b: &Tensor,
    rtol: Option<f64>,
    atol: Option<f64>,
    msg: Option<&str>,
) -> candle_core::Result<()> {
    let rtol = rtol.unwrap_or(1e-5);
    let atol = atol.unwrap_or(1e-8);

    if !allclose_with_tol(a, b, rtol, atol)? {
        let error_msg = msg.unwrap_or("Tensors are not close enough");
        panic!("{}", error_msg);
    }

    Ok(())
}

#[cfg(test)]
#[allow(dead_code)]
pub fn allclose(a: &Tensor, b: &Tensor) -> candle_core::Result<bool> {
    allclose_with_tol(a, b, 1e-5, 1e-8)
}

#[cfg(test)]
#[macro_export]
macro_rules! assert_tensor_close {
    ($a:expr, $b:expr) => {
        assert_allclose(&$a, &$b, None, None, None).expect("Tensors should be close")
    };
    ($a:expr, $b:expr, rtol=$rtol:expr) => {
        assert_allclose(&$a, &$b, Some($rtol), None, None).expect("Tensors should be close")
    };
    ($a:expr, $b:expr, atol=$atol:expr) => {
        assert_allclose(&$a, &$b, None, Some($atol), None).expect("Tensors should be close")
    };
    ($a:expr, $b:expr, rtol=$rtol:expr, atol=$atol:expr) => {
        assert_allclose(&$a, &$b, Some($rtol), Some($atol), None).expect("Tensors should be close")
    };
    ($a:expr, $b:expr, $msg:expr) => {
        assert_allclose(&$a, &$b, None, None, Some($msg)).expect("Tensors should be close")
    };
}
