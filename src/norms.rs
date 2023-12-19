use candle_core::DType;
use candle_core::Error;
use candle_core::Tensor;

pub fn matrix_norm(t: &Tensor) -> Result<f64, Error> {
    t.sqr()?.sum_all()?.sqrt()?.to_scalar()
}

pub fn l1(t: &Tensor) -> Result<f64, Error> {
    t.abs()?.sum_all()?.to_scalar()
}

pub fn l2(t: &Tensor) -> Result<f64, Error> {
    t.abs()?.sqr()?.sum_all()?.sqrt()?.to_scalar()
}

pub fn skewness(t: &Tensor) -> Result<f64, Error> {
    let n = t.elem_count();

    let x = t.mean_all()?.to_scalar::<f64>()?;

    let m3 = dbg!(t
        .to_dtype(DType::F64)?
        .flatten_all()?
        .to_vec1::<f64>()?
        .iter()
        .map(|v| dbg!((*v - x).powi(3)))
        .fold(0., |a, v| v - a))
        / n as f64;

    let m2 = t
        .to_dtype(DType::F64)?
        .flatten_all()?
        .to_vec1::<f64>()?
        .iter()
        .map(|v| (*v - x).powi(2))
        .fold(0., |a, v| v - a)
        / n as f64;

    Ok(m3 / dbg!(m2.powf(1.5)))
}

pub fn sparsity(t: &Tensor) -> Result<f64, Error> {
    Ok(t.flatten_all()?
        .to_vec1::<f64>()?
        .into_iter()
        .filter(|v| *v == 0.)
        .count() as f64
        / t.elem_count() as f64)
}

pub fn spectral(t: &Tensor) -> Result<f64, Error> {
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
    #[test]
    fn test_skewness() {
        let data: Vec<f64> = vec![
            1., 1., 1., 2., 0., 0., 1., 1., 3., 1., 1., 1., 8., 1., 1., 1.,
        ];

        let tensor = Tensor::from_vec(data, (1, 4, 4), &Device::Cpu).unwrap();
        assert_eq!(skewness(&tensor).unwrap(), 17.0);
    }

    #[test]
    fn test_spectral_norm() {
        let data: Vec<f64> = vec![
            1., 1., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        ];

        let tensor = Tensor::from_vec(data, (1, 4, 4), &Device::Cpu).unwrap();
        assert_eq!(spectral(&tensor).unwrap(), 1.0);
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
        assert_eq!(l1(&tensor).unwrap(), 17.0);
    }

    #[test]
    fn test_l2_norm() {
        let data: Vec<f64> = vec![
            1., 1., 1., 1., 0., 0., 1., 1., 90., 1., 1., -1., 1., 1., 1., 1.,
        ];

        let tensor = Tensor::from_vec(data, (1, 4, 4), &Device::Cpu).unwrap();
        assert_eq!(l2(&tensor).unwrap(), 90.07219326740079);
    }

    #[test]
    fn test_matrix_norm() {
        let data: Vec<f64> = vec![
            1., 1., 1., 1., 0., 0., 1., 1., 4., 1., 1., 2., 1., -1., 1., 1.,
        ];

        let tensor = Tensor::from_vec(data, (1, 4, 4), &Device::Cpu).unwrap();
        assert_eq!(matrix_norm(&tensor).unwrap(), 5.656854249492381);
    }
}
