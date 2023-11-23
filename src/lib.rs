use safetensors::{tensor::TensorInfo, SafeTensors};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

/// Helper struct used only for serialization deserialization
#[derive(Serialize, Deserialize)]
struct HashMetadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "__metadata__")]
    metadata: Option<HashMap<String, String>>,
    #[serde(flatten)]
    tensors: HashMap<String, TensorInfo>,
}

#[wasm_bindgen]
pub fn get_metadata_from_buffer(buffer: &[u8]) -> JsValue {
    console_error_panic_hook::set_once();

    match SafeTensors::read_metadata(buffer) {
        Ok((_, metadata)) => match metadata.metadata() {
            Some(metadata) => match serde_wasm_bindgen::to_value(metadata) {
                Ok(metadata) => metadata,
                Err(_) => todo!(),
            },
            None => match serde_wasm_bindgen::to_value(&HashMap::<String, String>::from([(
                "error".to_string(),
                "Could not find any metadata in this LoRA".to_string(),
            )])) {
                Ok(metadata) => metadata,
                Err(_) => todo!(),
            },
        },
        Err(_) => todo!(),
    }
}

// def get_vector_data_strength(data: dict[int, Tensor]) -> float:
//     value = 0
//     for n in data:
//         value += abs(n)
//
//     # the average value of each vector (ignoring negative values)
//     return value / len(data)
//
//
// # Euclidian norm
// def get_vector_data_magnitude(data: dict[int, Tensor]) -> float:
//     value = 0
//     for n in data:
//         value += pow(n, 2)
//     return math.sqrt(value)

// #[wasm_bindgen]
// pub fn get_average_magnitude(buffer: &[u8]) -> JsValue {
//     console_error_panic_hook::set_once();
//
//     let dtype = Dtype::F32;
//     match SafeTensors::deserialize(buffer) {
//         Ok(x) => {
//             //
//             // let n: usize = tensors_desc
//             // .iter()
//             // .map(|(_, shape)| shape.iter().product::<usize>())
//             // .sum::<usize>()
//             // * dtype.size(); // 4
//
//             let tensors = x.tensors().iter().map(|(k, v)| {
//                 // only check unet for the moment
//                 if k.contains("weight") && (k.contains("down") || k.contains("up")) {
//                     0
//                 } else {
//                     v.data().iter().product()
//                 }
//             }).collect();
//             match serde_wasm_bindgen::to_value(&tensors) {
//                 Ok(metadata) => metadata,
//                 Err(_) => todo!(),
//             }
//         }
//         Err(_) => todo!(),
//     }
// }
//
// pub fn get_average_strength(buffer: &[u8]) -> JsValue {
//     console_error_panic_hook::set_once();
//     match SafeTensors::deserialize(buffer) {
//         Ok(x) => {
//             //
//             let sum: f64 = x
//                 .tensors()
//                 .iter()
//                 .map(|(k, v)| {
//                     // only check unet for the moment
//                     if k.contains("weight") && (k.contains("down") || k.contains("up")) {
//                         0 as f64
//                     } else {
//                         v.data().iter().sum::<u8>() as f64
//                     }
//                 })
//                 .sum();
//             match serde_wasm_bindgen::to_value(&(sum / x.len() as f64)) {
//                 Ok(metadata) => metadata,
//                 Err(_) => todo!(),
//             }
//         }
//         Err(_) => todo!(),
//     }
// }

// pub fn get_tensor<'a>(
//     name: &str,
//     metadata: Metadata,
//     offset: usize,
// ) -> Result<SafeTensors<'a>, SafeTensorError> {
//     // let (n, metadata) = SafeTensors::read_metadata(&buffer).map_err(|e| {
//     //     SafetensorError::new_err(format!("Error while deserializing header: {e:?}"))
//     // })?;
//
//     // let offset = n + 8;
//     let info = metadata
//         .info(name)
//         .ok_or_else(|| SafeTensorError::new_err(format!("File does not contain tensor {name}",)))?;
//     // let dtype: PyObject = get_pydtype(torch, info.dtype, false)?;
//
//     // info.dtype
//     let shape = info.shape.to_vec();
//     let start = (info.data_offsets.0 + offset) as isize;
//     let stop = (info.data_offsets.1 + offset) as isize;
// }

// #[derive(Debug, Clone)]
// pub struct Metadata {
//     metadata: Option<HashMap<String, String>>,
//     tensors: Vec<TensorInfo>,
//     index_map: HashMap<String, usize>,
// }

// #[derive(Debug, Clone, PartialEq, Eq)]
// enum Device {
//     Cpu,
// }
//
// struct SafeSlice {
//     info: TensorInfo,
//     offset: usize,
//     device: Device,
// }
//
// fn slice_to_indexer(slice: &PySlice) -> Result<TensorIndexer, PyErr> {
//     let py_start = slice.getattr(intern!(slice.py(), "start"))?;
//     let start: Option<usize> = py_start.extract()?;
//     let start = if let Some(start) = start {
//         Bound::Included(start)
//     } else {
//         Bound::Unbounded
//     };
//
//     let py_stop = slice.getattr(intern!(slice.py(), "stop"))?;
//     let stop: Option<usize> = py_stop.extract()?;
//     let stop = if let Some(stop) = stop {
//         Bound::Excluded(stop)
//     } else {
//         Bound::Unbounded
//     };
//
//     Ok(TensorIndexer::Narrow(start, stop))
// }
//
// pub fn get_tensor_view() {
//     let attn_0 = TensorView::new(Dtype::F32, shape, &data).unwrap();
//     TensorView::new();
// }
//
// pub fn get_tensor_data(name: &str, buffer: &u8, metadata: Metadata, offset: usize) {
//     let info = metadata
//         .info(name)
//         .ok_or_else(|| SafeTensorError::new_err(format!("File does not contain tensor {name}",)))?;
//     // storage.as_ref();
//     let data = &buffer[info.data_offsets.0 + info.offset..info.data_offsets.1 + offset];
//
//     let tensor = TensorView::new(info.dtype, info.shape.clone(), data)
//         .map_err(|e| SafeTensorError::new_err(format!("Error preparing tensor view: {e:?}")))?;
//
//     // let slices: Vec<TensorIndexer> = slices
//     //     .into_iter()
//     //     .map(slice_to_indexer)
//     //     .collect::<Result<_, _>>()?;
//
//     let iterator = tensor.sliced_data(&slices).map_err(|e| {
//         SafeTensorError::new_err(format!(
//             "Error during slicing {slices:?} vs {:?}:  {:?}",
//             info.shape, e
//         ))
//     })?;
//     let newshape = iterator.newshape();
//
//     let mut offset = 0;
//     let length = iterator.remaining_byte_len();
//
//     for slice in iterator {
//         let len = slice.len();
//         bytes[offset..offset + slice.len()].copy_from_slice(slice);
//         offset += len;
//     }
//
//     // Python::with_gil(|py| {
//     // let array: PyObject = PyByteArray::new_with(py, length, |bytes: &mut [u8]| {
//     //     Ok(())
//     // })?
//     // .into_py(py);
//     create_tensor(
//         &self.framework,
//         self.info.dtype,
//         &newshape,
//         array,
//         &self.device,
//     )
//     // })
// }
