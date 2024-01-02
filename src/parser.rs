// struct {
// prefix:
// block
// block_num
//     down_block
//     up_block
//         num
// block_type
//     attentions
// block_num
// sub_block
//     transformer_block
// sub_block_num
//     attn2
// to
//     num
//
// }

use std::collections::HashMap;

use pest::{iterators::Pairs, Parser};

use crate::{KeyParser, Rule};

fn parse(key: &str) -> Result<Pairs<Rule>, pest::error::Error<Rule>> {
    let successful_parse = KeyParser::parse(Rule::key, key);
    successful_parse
    // if let Ok(pairs) = successful_parse {
    //     console::log_1(&format!("{:#?}", pairs).into());
    // }
}

// fn process(pair: Pair<Rule>) {}

fn process_ids() {
    let x = parse("lora_unet_down_blocks_0_attentions_0_proj_in").unwrap();

    let mut properties: HashMap<&str, u8> = HashMap::new();

    for pair in x {
        match pair.as_rule() {
            crate::Rule::block_id => {
                properties.insert("block_id", pair.as_str().parse::<u8>().unwrap());
            }
            crate::Rule::block_type_id => todo!(),
            crate::Rule::out_id => todo!(),
            crate::Rule::attn_id => todo!(),
            crate::Rule::layer_id => todo!(),
            _ => (),
        }
    }
}

#[cfg(test)]
mod tests {
    // use std::collections::HashMap;

    use super::parse;

    #[test]
    fn parse_key_test() {
        let x = parse("lora_unet_down_blocks_0_attentions_0_proj_in").unwrap();

        // let mut properties: HashMap<&str, HashMap<&str, &str>> = HashMap::new();

        // for pair in x {
        //     match pair.as_rule() {
        //         crate::Rule::block_id => todo!(),
        //         crate::Rule::block_type_id => todo!(),
        //         crate::Rule::out_id => todo!(),
        //         crate::Rule::attn_id => todo!(),
        //         crate::Rule::layer_id => todo!(),
        //     }
        // }

        println!("{:#?}", x.tokens());
    }
}
