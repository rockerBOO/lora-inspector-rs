use crate::{InspectorError, Result};
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
fn mean<T: Div + candle_core::WithDType + num::FromPrimitive>(t: &Tensor) -> Result<T> {
    let sum = t.sum_all()?.to_scalar::<T>()?;
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
        (Ok(t_mean), count) if count > 0 => Ok(variance(t, t_mean, 2, count)?.sqrt()),
        _ => Err(InspectorError::Msg(
            "Invalid std deviation calcualtion".to_owned(),
        )),
    }
}

pub fn variance<T>(t: &Tensor, mean: T, ordinal: i32, count: usize) -> Result<T>
where
    T: candle_core::WithDType + num::FromPrimitive + std::iter::Sum + num::traits::real::Real,
{
    Ok(t.flatten_all()?
        .to_vec1::<T>()?
        .iter()
        .map(|value| {
            // let diff = mean - *value;
            let diff = *value - mean;
            diff.powi(ordinal)
        })
        .sum::<T>()
        / FromPrimitive::from_usize(count).unwrap())
}

pub fn moment<T: num::Float + num::FromPrimitive>(distribution: &[T], mean: T, pow: i32) -> T {
    distribution
        .iter()
        .map(|v| (*v - mean).powi(pow))
        .fold(FromPrimitive::from_f64(0.).unwrap(), |a, v| v - a)
}


// Need to get working...
// pub fn skewness<T>(t: &Tensor) -> Result<T>
// where
//     T: num::Float + num::FromPrimitive + candle_core::WithDType,
// {
//     let n = t.elem_count();
//
//     let x = t.mean_all()?.to_scalar::<T>()?;
//     let distribution = t.flatten_all()?.to_vec1::<T>()?;
//
//     let m3 = moment(&distribution, x, 3) / FromPrimitive::from_usize(n).unwrap();
//     let m2 = moment(&distribution, x, 2) / FromPrimitive::from_usize(n).unwrap();
//
//     println!(
//         "{}",
//         -(2.75_f64.powf(FromPrimitive::from_f64(1.5).unwrap()))
//     );
//     println!("{}  {} ", m3, m2);
//
//     Ok(m3 / m2.powf(FromPrimitive::from_f64(1.5).unwrap()))
// }

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
        let data = vec![50, 32, 98, 46, 7];

        assert_eq!(data.get(1), select(&data, 1).as_ref());
    }

    #[test]
    fn test_select_outside() {
        let data = vec![50, 32, 98, 46, 7];

        assert_eq!(None, select(&data, 10));
    }

    #[test]
    fn test_median_unsorted() -> Result<()> {
        let device = Device::Cpu;
        let t = Tensor::new(&[5_i64, 2, 4, 3, 1], &device)?;

        assert_eq!(3, median::<i64>(&t).unwrap());

        Ok(())
    }

    #[test]
    fn test_median_even() -> Result<()> {
        let device = Device::Cpu;
        let t = Tensor::new(&[2_f64, 1.], &device)?;

        assert_eq!(1.5, median::<f64>(&t).unwrap());

        Ok(())
    }

    #[test]
    fn test_median_odd() -> Result<()> {
        let device = Device::Cpu;
        let t = Tensor::new(&[3_i64, 1], &device)?;

        assert_eq!(2, median::<i64>(&t)?);

        Ok(())
    }

    #[test]
    fn test_mean() -> Result<()> {
        let device = Device::Cpu;
        let t = Tensor::new(&[1.0_f32, 2.0, 3.0, 4.0], &device)?;

        assert_eq!(2.5, mean::<f32>(&t).unwrap());

        Ok(())
    }

    #[test]
    fn test_mean_single() -> Result<()> {
        let device = Device::Cpu;
        let t = Tensor::new(&[1.0_f32], &device)?; // replace with your actual data
        assert_eq!(1.0, mean::<f32>(&t).unwrap());

        Ok(())
    }
}
