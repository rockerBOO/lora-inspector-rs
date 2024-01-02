use crate::Result;
use candle_core::Tensor;

pub struct NormFn<T: candle_core::WithDType> {
    pub name: String,
    pub function: Box<dyn Fn(Tensor) -> Result<T>>,
}

impl<T: candle_core::WithDType> std::ops::Deref for NormFn<T> {
    type Target = dyn Fn(Tensor) -> Result<T>;

    fn deref(&self) -> &Self::Target {
        &self.function
    }
}

pub fn matrix_norm<T>(t: &Tensor) -> Result<T>
where
    T: candle_core::WithDType,
{
    Ok(t.sqr()?.sum_all()?.sqrt()?.to_scalar()?)
}

pub fn l1<T>(t: &Tensor) -> Result<T>
where
    T: candle_core::WithDType,
{
    Ok(t.abs()?.sum_all()?.to_scalar()?)
}

pub fn l2<T>(t: &Tensor) -> Result<T>
where
    T: candle_core::WithDType,
{
    Ok(t.abs()?.sqr()?.sum_all()?.sqrt()?.to_scalar()?)
}

pub fn sparsity(t: &Tensor) -> Result<f64> {
    Ok(t.flatten_all()?
        .to_vec1::<f64>()?
        .into_iter()
        .filter(|v| *v as u64 == 0)
        .count() as f64
        / t.elem_count() as f64)
}

pub fn max(t: &Tensor) -> Result<f64> {
    Ok(t.flatten_all()?.max(0)?.to_dtype(candle_core::DType::F64)?.to_scalar::<f64>()?)
}

pub fn min(t: &Tensor) -> Result<f64> {
    Ok(t.flatten_all()?.min(0)?.to_dtype(candle_core::DType::F64)?.to_scalar::<f64>()?)
}

pub fn spectral(t: &Tensor) -> Result<f64> {
    Ok(t.flatten_all()?
        .to_vec1::<f64>()?
        .into_iter()
        .reduce(f64::max)
        .unwrap_or(0.))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};
    // #[test]
    // fn test_skewness() {
    //     let data: Vec<f64> = vec![
    //         1., 1., 1., 2., 0., 0., 1., 1., 3., 1., 1., 1., 8., 1., 1., 1.,
    //     ];
    //
    //     let tensor = Tensor::from_vec(data, (1, 4, 4), &Device::Cpu).unwrap();
    //     assert_eq!(skewness::<f64>(&tensor).unwrap(), 2.8801740957848434);
    // }

    #[test]
    fn test_spectral_norm() {
        let data: Vec<f64> = vec![
            1., 1., 1., 1., 0., 0., 1., 1., 1., 2., 9., 1., 1., -2., 1., 1.,
        ];

        let tensor = Tensor::from_vec(data, (1, 4, 4), &Device::Cpu).unwrap();
        assert_eq!(spectral(&tensor).unwrap(), 9.);
    }

    #[test]
    fn test_sparsity() {
        let data: Vec<f64> = vec![
            1., 1., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        ];

        let tensor = Tensor::from_vec(data, (1, 4, 4), &Device::Cpu).unwrap();
        assert_eq!(sparsity(&tensor).unwrap(), 0.125);
    }

    #[test]
    fn test_l1_norm() {
        let data: Vec<f64> = vec![
            1., 1., 1., 2., 0., 0., 1., 1., 3., 1., 1., 1., 1., 1., 1., 1.,
        ];

        let tensor = Tensor::from_vec(data, (1, 4, 4), &Device::Cpu).unwrap();
        assert_eq!(l1::<f64>(&tensor).unwrap(), 17.0);
    }

    #[test]
    fn test_l2_norm() {
        let data: Vec<f64> = vec![
            1., 1., 1., 1., 0., 0., 1., 1., 90., 1., 1., -1., 1., 1., 1., 1.,
        ];

        let tensor = Tensor::from_vec(data, (1, 4, 4), &Device::Cpu).unwrap();
        assert_eq!(l2::<f64>(&tensor).unwrap(), 90.07219326740079);
    }

    #[test]
    fn test_matrix_norm() {
        let data: Vec<f64> = vec![
            1., 1., 1., 1., 0., 0., 1., 1., 4., 1., 1., 2., 1., -1., 1., 1.,
        ];

        let tensor = Tensor::from_vec(data, (1, 4, 4), &Device::Cpu).unwrap();
        assert_eq!(matrix_norm::<f64>(&tensor).unwrap(), 5.656854249492381);
    }
}
