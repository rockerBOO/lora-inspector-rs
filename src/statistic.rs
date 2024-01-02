use candle_core::Tensor;
use num::FromPrimitive;
use num::NumCast;
use std::cmp::Ordering;
use std::ops::Div;

use crate::InspectorError;
use crate::Result;

// https://rust-lang-nursery.github.io/rust-cookbook/science/mathematics/statistics.html
pub fn partition<T>(data: &[T]) -> Option<(Vec<T>, T, Vec<T>)>
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
pub fn select<T: Div + std::cmp::PartialOrd + std::marker::Copy>(
    data: &[T],
    k: usize,
) -> Option<T> {
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
pub fn median<T: NumCast + Div + candle_core::WithDType + num::FromPrimitive>(
    t: &Tensor,
) -> Result<T> {
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
pub fn mean<T: Div + candle_core::WithDType + num::FromPrimitive>(t: &Tensor) -> Result<T> {
    let sum = t.sum_all()?.to_scalar::<T>()?;
    let count = t.elem_count();

    match count {
        positive if positive > 0 => Ok(sum / FromPrimitive::from_usize(count).unwrap()),
        _ => Err(InspectorError::Msg("Invalid mean calculation".to_owned())),
    }
}

// https://rust-lang-nursery.github.io/rust-cookbook/science/mathematics/statistics.html
pub fn std_deviation<
    T: Div
        + candle_core::WithDType
        // + std::borrow::Borrow<Tensor>
        + num::FromPrimitive
        + num::Float
        + std::iter::Sum
        + std::fmt::Debug,
>(
    t: &Tensor,
) -> Result<Option<T>> {
    match (mean::<T>(t), t.elem_count()) {
        (Ok(t_mean), count) if count > 0 => Ok(Some(variance(t, t_mean, 2, count)?.sqrt())),
        (Err(e), _count) => Err(e),
        _ => Ok(None),
    }
}

pub fn variance<T>(t: &Tensor, mean: T, ordinal: i32, count: usize) -> Result<T>
where
    T: candle_core::WithDType
        + num::FromPrimitive
        + std::iter::Sum
        + num::traits::real::Real
        + std::fmt::Debug,
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

#[cfg(test)]
mod tests {
    use super::*;

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
        let device = candle_core::Device::Cpu;
        let t = Tensor::new(&[5_i64, 2, 4, 3, 1], &device)?;

        assert_eq!(3, median::<i64>(&t).unwrap());

        Ok(())
    }

    #[test]
    fn test_median_even() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let t = Tensor::new(&[2_f64, 1.], &device)?;

        assert_eq!(1.5, median::<f64>(&t).unwrap());

        Ok(())
    }

    #[test]
    fn test_median_odd() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let t = Tensor::new(&[3_i64, 1], &device)?;

        assert_eq!(2, median::<i64>(&t)?);

        Ok(())
    }

    #[test]
    fn test_mean() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let t = Tensor::new(&[1.0_f32, 2.0, 3.0, 4.0], &device)?;
        assert_eq!(2.5, mean::<f32>(&t).unwrap());

        let t = Tensor::new(&[1.0_f32, 2.0, 3.0], &device)?;
        assert_eq!(2., mean::<f32>(&t).unwrap());

        Ok(())
    }

    #[test]
    fn test_mean_single() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let t = Tensor::new(&[1.0_f32], &device)?; // replace with your actual data
        assert_eq!(1.0, mean::<f32>(&t).unwrap());

        Ok(())
    }

    #[test]
    fn test_std_deviation() {
        let device = candle_core::Device::Cpu;
        let t = Tensor::new(&[1_f64, 2., 3.], &device).unwrap();

        assert_eq!(std_deviation(&t).unwrap(), Some(0.816496580927726)); // Expected output is the standard deviation of [1,2,3] which approximately equals to 0.816...
    }
}
