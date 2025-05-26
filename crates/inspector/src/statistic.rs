use candle_core::Tensor;
use num::FromPrimitive;
use num::NumCast;
// use std::cmp::Ordering;
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

// // https://rust-lang-nursery.github.io/rust-cookbook/science/mathematics/statistics.html
// pub fn select<T: Div + std::cmp::PartialOrd + std::marker::Copy>(
//     data: &[T],
//     k: usize,
// ) -> Option<T> {
//     let part = partition(data);
//
//     match part {
//         None => None,
//         Some((left, pivot, right)) => {
//             let pivot_idx = left.len();
//
//             match pivot_idx.cmp(&k) {
//                 Ordering::Equal => Some(pivot),
//                 Ordering::Greater => select(&left, k),
//                 Ordering::Less => select(&right, k - (pivot_idx + 1)),
//             }
//         }
//     }
// }
//
// // https://rust-lang-nursery.github.io/rust-cookbook/science/mathematics/statistics.html
// pub fn median<T: NumCast + Div + candle_core::WithDType + num::FromPrimitive>(
//     t: &Tensor,
// ) -> Result<T> {
//     let size = t.elem_count();
//     let data = t.flatten_all()?.to_vec1::<T>()?;
//
//     match size {
//         even if even % 2 == 0 => {
//             let fst_med = select(&data, (even / 2) - 1);
//             let snd_med = select(&data, even / 2);
//
//             match (fst_med, snd_med) {
//                 (Some(fst), Some(snd)) => Ok((fst + snd) / FromPrimitive::from_usize(2).unwrap()),
//                 _ => Err(InspectorError::Msg("could not process median".to_owned())),
//             }
//         }
//         odd => select(&data, odd / 2)
//             .ok_or_else(|| InspectorError::Msg("Could not get odd calculation".to_owned())),
//     }
// }
//
// // https://rust-lang-nursery.github.io/rust-cookbook/science/mathematics/statistics.html
// pub fn mean<T: Div + candle_core::WithDType + num::FromPrimitive>(t: &Tensor) -> Result<T> {
//     let sum = t.sum_all()?.to_scalar::<T>()?;
//     let count = t.elem_count();
//
//     match count {
//         positive if positive > 0 => Ok(sum / FromPrimitive::from_usize(count).unwrap()),
//         _ => Err(InspectorError::Msg("Invalid mean calculation".to_owned())),
//     }
// }

// Mean function updated to work directly on tensors
pub fn mean<T: Div + candle_core::WithDType + num::FromPrimitive>(t: &Tensor) -> Result<T> {
    let sum = t.sum_all()?.to_scalar::<T>()?;
    let count = t.elem_count();
    match count {
        positive if positive > 0 => Ok(sum / FromPrimitive::from_usize(count).unwrap()),
        _ => Err(InspectorError::Msg("Invalid mean calculation".to_owned())),
    }
}

// Variance function working directly on tensors
pub fn variance<T>(t: &Tensor, mean_val: Option<T>) -> Result<T>
where
    T: candle_core::WithDType
        + num::FromPrimitive
        + std::iter::Sum
        + num::traits::real::Real
        + std::fmt::Debug,
{
    let flat = t.flatten_all()?;
    let count = flat.elem_count();

    // Use provided mean or calculate it
    let mean = match mean_val {
        Some(val) => val,
        None => mean::<T>(&flat)?,
    };

    // Create a scalar tensor with the mean value and broadcast it
    let mean_tensor = Tensor::new(&[mean], flat.device())?
        .to_dtype(flat.dtype())?
        .broadcast_as(flat.shape())?;

    // Calculate squared differences
    let diff_squared = (&flat - &mean_tensor)?.sqr()?;

    // Get sum and calculate variance
    let sum = diff_squared.sum_all()?.to_scalar::<T>()?;
    let variance = sum / FromPrimitive::from_usize(count).unwrap();

    Ok(variance)
}

// Standard deviation function using the variance function
pub fn std_dev<T>(t: &Tensor) -> Result<T>
where
    T: candle_core::WithDType
        + num::FromPrimitive
        + std::iter::Sum
        + num::traits::real::Real
        + std::fmt::Debug,
{
    let var = variance::<T>(t, None)?;
    Ok(var.sqrt())
}

// Median function - this one is trickier as we need sorting functionality
// We'll implement a tensor-based approach if possible, otherwise fall back to the select method
pub fn median<
    T: NumCast + Div + candle_core::WithDType + num::FromPrimitive + PartialOrd + Copy,
>(
    t: &Tensor,
) -> Result<T> {
    // For now, we need to convert to vec and sort
    let mut data = t.flatten_all()?.to_vec1::<T>()?;

    // Sort the data
    data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let size = data.len();
    match size {
        0 => Err(InspectorError::Msg(
            "Cannot compute median of empty tensor".to_owned(),
        )),
        even if even % 2 == 0 => {
            let mid = size / 2;
            let median = (data[mid - 1] + data[mid]) / FromPrimitive::from_f64(2.0).unwrap();
            Ok(median)
        }
        _ => Ok(data[size / 2]),
    }
}

pub fn median_efficient<T>(t: &Tensor) -> Result<T>
where
    T: candle_core::WithDType + num::FromPrimitive + PartialOrd + Copy + Div<Output = T> + NumCast,
{
    // For small tensors, just use the regular approach
    if t.elem_count() < 10000 {
        return median::<T>(t);
    }

    let flat = t.flatten_all()?;
    let n = flat.elem_count();

    // Find min and max to establish bounds
    let min_val = flat.min(0)?.to_scalar::<T>()?;
    let max_val = flat.max(0)?.to_scalar::<T>()?;

    // Prepare for binary search
    let target_count = if n % 2 == 0 { n / 2 } else { n / 2 + 1 };
    let mut low = min_val;
    let mut high = max_val;
    let mut mid_val: T;

    // Epsilon for floating point comparison
    let epsilon: T = NumCast::from(1e-10).unwrap_or_else(|| NumCast::from(0).unwrap());

    // Binary search to find median value
    while (high - low) > epsilon {
        // Calculate midpoint - avoiding potential overflow
        mid_val = low + (high - low) / NumCast::from(2.0).unwrap();

        // Count elements less than or equal to mid_val
        // This is the key part that uses tensor operations
        let count = count_less_equal(&flat, mid_val)?;

        if count < target_count {
            low = mid_val;
        } else {
            high = mid_val;
        }

        // Break early if we're close enough
        if (high - low) <= epsilon {
            break;
        }
    }

    // Handle even length arrays (average of two middle values)
    if n % 2 == 0 {
        // For even length, we need one more value
        // Find the largest value smaller than our current median
        let lower_median = find_lower_median(&flat, low, min_val)?;
        let median = (low + lower_median) / NumCast::from(2.0).unwrap();
        Ok(median)
    } else {
        Ok(low) // For odd length, the current value is the median
    }
}

// Count elements less than or equal to a value using tensor operations
fn count_less_equal<T>(tensor: &Tensor, value: T) -> Result<usize>
where
    T: candle_core::WithDType + Copy,
{
    // Create a tensor with the value
    let value_tensor = Tensor::new(&[value], tensor.device())?
        .to_dtype(tensor.dtype())?
        .broadcast_as(tensor.shape())?;

    // Create a mask where elements <= value are 1, others are 0
    let mask = tensor.le(&value_tensor)?;

    // Sum the mask to get the count
    let count = mask.sum_all()?.to_scalar::<u32>()?;

    Ok(count as usize)
}

// Function to find the largest value smaller than or equal to the median
// This is needed for computing the median of even-length arrays
fn find_lower_median<T>(tensor: &Tensor, median: T, min_val: T) -> Result<T>
where
    T: candle_core::WithDType + num::FromPrimitive + PartialOrd + Copy + Div<Output = T> + NumCast,
{
    // Handle the case where median is the minimum value
    if median <= min_val {
        return Ok(median);
    }

    // Create a mask for elements < median
    let value_tensor = Tensor::new(&[median], tensor.device())?
        .to_dtype(tensor.dtype())?
        .broadcast_as(tensor.shape())?;

    let mask = tensor.lt(&value_tensor)?;

    // Use the mask to extract values less than median
    let smaller_elements = masked_select::<f64>(tensor, &mask)?;

    // If no smaller elements, return the minimum
    if smaller_elements.elem_count() == 0 {
        return Ok(min_val);
    }

    // Get maximum of these smaller elements
    let lower_median = smaller_elements.max(0)?.to_scalar::<T>()?;

    Ok(lower_median)
}

/// Select elements from a tensor based on a boolean mask represented as U8
pub fn masked_select<T>(tensor: &Tensor, mask: &Tensor) -> Result<Tensor>
where
    T: candle_core::WithDType + Copy,
{
    // Ensure mask is U8 type
    let bool_mask = if mask.dtype() == candle_core::DType::U8 {
        mask.clone()
    } else {
        // Convert to U8 type if needed
        mask.to_dtype(candle_core::DType::U8)?
    };

    // Flatten both tensors
    let flat_tensor = tensor.flatten_all()?;
    let flat_mask = bool_mask.flatten_all()?;

    // Check if shapes match after flattening
    if flat_tensor.shape() != flat_mask.shape() {
        return Err(InspectorError::Msg(format!(
            "Tensor shape {:?} and mask shape {:?} don't match after flattening",
            flat_tensor.shape(),
            flat_mask.shape()
        )));
    }

    // Convert tensors to host for processing
    let tensor_vec = flat_tensor.to_vec1::<T>()?;
    let mask_vec = flat_mask.to_vec1::<u8>()?;

    // Select elements where mask is non-zero (true)
    let mut result = Vec::new();
    for (val, &mask_val) in tensor_vec.iter().zip(mask_vec.iter()) {
        if mask_val != 0 {
            result.push(*val);
        }
    }

    // Create a new tensor from the selected values
    Ok(Tensor::new(result.as_slice(), tensor.device())?.to_dtype(tensor.dtype())?)
}

// // https://rust-lang-nursery.github.io/rust-cookbook/science/mathematics/statistics.html
// pub fn std_deviation<
//     T: Div
//         + candle_core::WithDType
//         // + std::borrow::Borrow<Tensor>
//         + num::FromPrimitive
//         + num::Float
//         + std::iter::Sum
//         + std::fmt::Debug,
// >(
//     t: &Tensor,
// ) -> Result<Option<T>> {
//     match (mean::<T>(t), t.elem_count()) {
//         (Ok(t_mean), count) if count > 0 => Ok(Some(variance(t, t_mean, 2, count)?.sqrt())),
//         (Err(e), _count) => Err(e),
//         _ => Ok(None),
//     }
// }

// pub fn variance<T>(t: &Tensor, mean: T, ordinal: i32, count: usize) -> Result<T>
// where
//     T: candle_core::WithDType
//         + num::FromPrimitive
//         + std::iter::Sum
//         + num::traits::real::Real
//         + std::fmt::Debug,
// {
//     Ok(t.flatten_all()?
//         .to_vec1::<T>()?
//         .iter()
//         .map(|value| {
//             // let diff = mean - *value;
//             let diff = *value - mean;
//             diff.powi(ordinal)
//         })
//         .sum::<T>()
//         / FromPrimitive::from_usize(count).unwrap())
// }
// pub fn variance<T>(t: &Tensor, mean: T, ordinal: i32, count: usize) -> Result<T>
// where
//     T: candle_core::WithDType
//         + num::FromPrimitive
//         + std::iter::Sum
//         + num::traits::real::Real
//         + std::fmt::Debug,
// {
//     let flat = t.flatten_all()?;
//
//     // Create a scalar tensor with the mean value
//     let mean_scalar = Tensor::new(&[mean], flat.device())?
//         .to_dtype(flat.dtype())?;
//
//     // Broadcast the mean to match the shape of flat
//     // Using the broadcast_as method which will handle the shape difference
//     let mean_tensor = mean_scalar.broadcast_as(flat.shape())?;
//
//     // Calculate the difference
//     let diff = (&flat - &mean_tensor)?;
//
//     // Apply the power operation
//     let powered = match ordinal {
//         1 => diff.clone(),
//         2 => diff.sqr()?,
//         _ => {
//             // For other powers, we need to go through element-wise operations
//             let flat_vec = diff.to_vec1::<T>()?;
//             let powered_vec: Vec<T> = flat_vec.iter()
//                 .map(|&x| x.powi(ordinal))
//                 .collect();
//             Tensor::new(powered_vec.as_slice(), flat.device())?.to_dtype(flat.dtype())?
//         }
//     };
//
//     // Calculate the sum and divide by count
//     let sum = powered.sum_all()?;
//     let sum_val = sum.to_scalar::<T>()?;
//     let variance = sum_val / FromPrimitive::from_usize(count).unwrap();
//
//     Ok(variance)
// }

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

    // #[test]
    // fn test_select() {
    //     let data: Vec<i64> = vec![5, 3, 2, 1];
    //
    //     assert_eq!(data.get(3), select(&data, 0).as_ref());
    //     assert_eq!(None, select::<i64>(&[], 0));
    // }
    //
    // #[test]
    // fn test_select_middle() {
    //     let data = vec![5, 3, 2, 1]; // replace with your actual data
    //
    //     assert_eq!(Some(2), select(&data, 1));
    // }
    //
    // #[test]
    // fn test_select_large() {
    //     let data = vec![50, 32, 98, 46, 7];
    //
    //     assert_eq!(data.get(1), select(&data, 1).as_ref());
    // }
    //
    // #[test]
    // fn test_select_outside() {
    //     let data = vec![50, 32, 98, 46, 7];
    //
    //     assert_eq!(None, select(&data, 10));
    // }

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

        assert_eq!(std_dev::<f64>(&t).unwrap(), 0.816496580927726_f64); // Expected output is the standard deviation of [1,2,3] which approximately equals to 0.816...
    }
}
