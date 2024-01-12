use std::{fs::File, io::Read};
use inspector::{Result, file, metadata};

fn main() -> Result<()> {
    let filename = "boo.safetensors";

    let mut f = File::open(filename)?;
    let mut data = vec![];
    f.read_to_end(&mut data)?;

    let metadata = metadata::Metadata::new_from_buffer(data.as_slice()).map_err(|e| e.to_string());
    let file = file::LoRAFile::new_from_buffer(data.as_slice(), filename);

    // metadata.map(|metadata| LoraWorker { metadata, file })
    
    Ok(())
}
