use candle_core::{Device, Tensor};
use memmap2::MmapOptions;
use safetensors::SafeTensors;
use std::fs::File;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let filename = "/mnt/900/training/sets/pov-2023-11-25-025803-4cf6f9ce/pov-2023-11-25-025803-4cf6f9ce.safetensors";
    let file = File::open(filename).unwrap();
    let buffer = unsafe { MmapOptions::new().map(&file).unwrap() };

    let (_, metadata) = SafeTensors::read_metadata(&buffer).unwrap();

    let metadata = match metadata.metadata() {
        Some(metadata) => metadata,
        None => todo!(),
    };

    for (k, v) in metadata {
        println!("{k} {v}");
    }

    let device = Device::Cpu;

    let a = Tensor::randn(0f32, 1., (2, 3), &device)?;
    let b = Tensor::randn(0f32, 1., (3, 4), &device)?;

    let c = a.matmul(&b)?;
    println!("{c}");
    Ok(())

    // let tensors = SafeTensors::deserialize(&buffer).unwrap();
}
