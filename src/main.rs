use memmap2::MmapOptions;
use safetensors::SafeTensors;
use std::fs::File;
fn main() {
    let filename = "64c92856.safetensors";
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

    // let tensors = SafeTensors::deserialize(&buffer).unwrap();
}
