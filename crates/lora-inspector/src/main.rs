use inspector::{file, metadata, norms, Result};
use std::{fs::File, io::Read};

fn main() -> Result<()> {
    let filename = "boo.safetensors";
    let device = candle_core::Device::Cpu;

    let mut f = File::open(filename)?;
    let mut data = vec![];
    f.read_to_end(&mut data)?;

    let metadata = metadata::Metadata::new_from_buffer(data.as_slice()).map_err(|e| e.to_string());
    let mut file = file::LoRAFile::new_from_buffer(data.as_slice(), filename);

    let base_names = file.base_names();

    base_names.iter().for_each(|base_name| {
        // println!("{:?} {:?}", base_name, norms::l2::<f32>(
        //     &file
        //         .scale_weight(base_name, &device)
        //         .unwrap()
        //         .to_dtype(candle_core::DType::F32)
        //         .unwrap(),
        // )
        // .unwrap());
        // "l1_norm",
        // "l2_norm",
        // "matrix_norm",
        // "max",
        // "min",
        // "std_dev",
        // "median",
        let scale_weight = file.scale_weight(base_name, &device).unwrap().to_dtype(candle_core::DType::F64).unwrap();
        println!(
            "{:?} {:?}",
            base_name,
            norms::l2::<f64>(&scale_weight)
        );
        println!("{:?}", norms::l1::<f64>(&scale_weight));
        println!("{:?}", norms::matrix_norm::<f64>(&scale_weight));
        println!("{:?}", norms::max(&scale_weight));
        println!(
            "{:?}",
            norms::min(&file.scale_weight(base_name, &device).unwrap())
        );
    });

    println!("{:?}", base_names);

    // metadata.map(|metadata| LoraWorker { metadata, file })
    println!("Hello");

    Ok(())
}
