mod file;
mod metadata;
mod network;
mod norms;
mod weight;
mod worker;

#[derive(Debug)]
pub enum Error {
    Candle(candle_core::Error),
    SafeTensor(safetensors::SafeTensorError),
    Load(String),
    Msg(String),
}

pub fn get_base_name(name: &str) -> String {
    name.split('.')
        .filter(|part| !matches!(*part, "weight" | "lora_up" | "lora_down" | "alpha"))
        .fold(String::new(), |acc, v| {
            if acc.is_empty() {
                v.to_owned()
            } else {
                format!("{acc}.{v}")
            }
        })
}

#[cfg(test)]
mod tests {
    use crate::get_base_name;

    #[test]
    fn get_base_name_test() {
        let base_name = get_base_name("lora_unet_up_blocks_1_attentions_1_proj_out.lora_up.weight");
        assert_eq!(base_name, "lora_unet_up_blocks_1_attentions_1_proj_out");

        let base_name =
            get_base_name("lora_unet_up_blocks_1_attentions_1_proj_out.lora_down.weight");
        assert_eq!(base_name, "lora_unet_up_blocks_1_attentions_1_proj_out");

        let base_name = get_base_name("lora_unet_up_blocks_1_attentions_1_proj_out.alpha");
        assert_eq!(base_name, "lora_unet_up_blocks_1_attentions_1_proj_out");
    }
}
