use crate::{InspectorError, Result};
use candle_core::DType;
use candle_core::Tensor;
use num::FromPrimitive;
use num::NumCast;
use std::cmp::Ordering;
use std::ops::Div;

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

// https://rust-lang-nursery.github.io/rust-cookbook/science/mathematics/statistics.html
fn partition<T>(data: &[T]) -> Option<(Vec<T>, T, Vec<T>)>
where
    T: Div + std::cmp::PartialOrd + Copy,
{
    match data.len() {
        0 => None,
        _ => {
            let (pivot_slice, tail) = data.split_at(1);
            let pivot = pivot_slice[0];
            let (left, right) = tail.iter().fold((vec![], vec![]), |mut splits, next| {
                {
                    let (ref mut left, ref mut right) = &mut splits;
                    if next < &pivot {
                        left.push(*next);
                    } else {
                        right.push(*next);
                    }
                }
                splits
            });

            Some((left, pivot, right))
        }
    }
}

// https://rust-lang-nursery.github.io/rust-cookbook/science/mathematics/statistics.html
fn select<T: Div + std::cmp::PartialOrd + std::marker::Copy>(data: &[T], k: usize) -> Option<T> {
    let part = partition(data);

    match part {
        None => None,
        Some((left, pivot, right)) => {
            let pivot_idx = left.len();

            match pivot_idx.cmp(&k) {
                Ordering::Equal => Some(pivot),
                Ordering::Greater => select(&left, k),
                Ordering::Less => select(&right, k - (pivot_idx + 1)),
            }
        }
    }
}

// https://rust-lang-nursery.github.io/rust-cookbook/science/mathematics/statistics.html
fn median<T: NumCast + Div + candle_core::WithDType + num::FromPrimitive>(t: &Tensor) -> Result<T> {
    let size = t.elem_count();
    let data = t.flatten_all()?.to_vec1::<T>()?;

    match size {
        even if even % 2 == 0 => {
            let fst_med = select(&data, (even / 2) - 1);
            let snd_med = select(&data, even / 2);

            match (fst_med, snd_med) {
                (Some(fst), Some(snd)) => Ok((fst + snd) / FromPrimitive::from_usize(2).unwrap()),
                _ => Err(InspectorError::Msg("could not process median".to_owned())),
            }
        }
        odd => select(&data, odd / 2)
            .ok_or_else(|| InspectorError::Msg("Could not get odd calculation".to_owned())),
    }
}

// https://rust-lang-nursery.github.io/rust-cookbook/science/mathematics/statistics.html
fn mean<
    T: Div + candle_core::WithDType + std::borrow::Borrow<candle_core::Tensor> + num::FromPrimitive,
>(
    t: &Tensor,
) -> Result<T> {
    let sum: T = t.sum_all()?.to_scalar()?;
    let count = t.elem_count();

    match count {
        positive if positive > 0 => Ok(sum / FromPrimitive::from_usize(count).unwrap()),
        _ => Err(InspectorError::Msg("Invalid mean calculation".to_owned())),
    }
}

// https://rust-lang-nursery.github.io/rust-cookbook/science/mathematics/statistics.html
fn std_deviation<
    T: Div
        + candle_core::WithDType
        + std::borrow::Borrow<candle_core::Tensor>
        + num::FromPrimitive
        + num::Float
        + std::iter::Sum,
>(
    t: &Tensor,
) -> Result<T> {
    match (mean::<T>(t), t.elem_count()) {
        (Ok(t_mean), count) if count > 0 => {
            let variance: T = t
                .flatten_all()?
                .to_vec1::<T>()?
                .iter()
                .map(|value| {
                    let diff = t_mean - *value;
                    diff * diff
                })
                .sum::<T>()
                / FromPrimitive::from_usize(count).unwrap();

            Ok(variance.sqrt())
        }
        _ => Err(InspectorError::Msg(
            "Invalid std deviation calcualtion".to_owned(),
        )),
    }
}

pub fn skewness(t: &Tensor) -> Result<f64> {
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

pub fn sparsity(t: &Tensor) -> Result<f64> {
    Ok(t.flatten_all()?
        .to_vec1::<f64>()?
        .into_iter()
        .filter(|v| *v == 0.)
        .count() as f64
        / t.elem_count() as f64)
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

    #[test]
    fn test_select() {
        let data: Vec<i64> = vec![5, 3, 2, 1];

        assert_eq!(data.get(3), select(&data, 0).as_ref());
        assert_eq!(None, select::<i64>(&[], 0));
    }

    #[test]
    fn test_select_middle() {
        let data = vec![5, 3, 2, 1]; // replace with your actual data

        assert_eq!(Some(2), select(&data, 1));
    }

    #[test]
    fn test_select_large() {
        let data = vec![50, 32, 98, 46, 7]; // replace with your actual large size vector.

        assert_eq!(data.get(1), select(&data, 1).as_ref());
    }

    #[test]
    fn test_select_outside() {
        let data = vec![50, 32, 98, 46, 7]; // replace with your actual large size vector.

        assert_eq!(None, select(&data, 10));
    }

    #[test]
    fn test_median_unsorted() -> Result<()> {
        let device = Device::Cpu;
        let t = Tensor::new(&[5_i64, 2, 4, 3, 1], &device)?; // replace with your actual data

        assert_eq!(3, median::<i64>(&t).unwrap());

        Ok(())
    }

    #[test]
    fn test_median_even() -> Result<()> {
        let device = Device::Cpu;
        let t = Tensor::new(&[2_f64, 1.], &device)?; // replace with your actual data

        assert_eq!(1.5, median::<f64>(&t).unwrap());

        Ok(())
    }

    #[test]
    fn test_median_odd() -> Result<()> {
        let device = Device::Cpu;
        let t = Tensor::new(&[3_i64, 1], &device)?; // replace with your actual data

        assert_eq!(2, median::<i64>(&t)?);

        Ok(())
    }
}
