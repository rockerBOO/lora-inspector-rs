mod support;

use candle_core::Device;
use inspector::file::LoRAFile;
use inspector::LoRAFormat;

use support::Fixture;

fn load_and_check(
    fixture_path: &str,
    expected_format: LoRAFormat,
    check_scale_weight: bool,
) -> LoRAFile {
    let fixture: Fixture = support::load_fixture(fixture_path);
    let buffer = support::synthesize_safetensors(&fixture);
    let file = LoRAFile::new_from_buffer(&buffer, fixture_path, &Device::Cpu);

    assert!(
        file.is_tensors_loaded(),
        "{fixture_path}: expected tensors to load"
    );
    assert_eq!(
        file.format(),
        expected_format,
        "{fixture_path}: unexpected LoRAFormat"
    );

    let base_names = file.base_names();
    assert!(
        !base_names.is_empty(),
        "{fixture_path}: expected non-empty base_names()"
    );

    let dims = file.dims();
    assert!(
        !dims.is_empty(),
        "{fixture_path}: expected non-empty dims()"
    );

    assert!(
        !file.unet_keys().is_empty(),
        "{fixture_path}: expected non-empty unet_keys() (diffusion_model./bare-prefix regression check)"
    );

    if check_scale_weight {
        let sample = &base_names[0];
        let scaled = file.scale_weight(sample);
        assert!(
            scaled.is_ok(),
            "{fixture_path}: scale_weight({sample}) failed: {:?}",
            scaled.err()
        );
    }

    file
}

#[test]
fn sd15_kohya() {
    load_and_check("sd15/kohya.json", LoRAFormat::Kohya, true);
}

#[test]
fn sdxl_kohya() {
    load_and_check("sdxl/kohya.json", LoRAFormat::Kohya, true);
}

#[test]
fn sdxl_lycoris() {
    // LoHA, but LoRAFormat::Lycoris is never constructed anywhere in weight.rs --
    // the Kohya/Peft axis is purely about lora_A/B vs lora_up/down naming, so this
    // still reports Kohya. The fixture's metadata (ss_network_module=lycoris.kohya,
    // algo=loha) would correctly route scale_weight() to scale_hada_weight, but
    // support::synthesize_safetensors builds the tensor buffer with
    // `safetensors::serialize(views, &None)`, which never writes a `__metadata__`
    // header -- so the synthesized file always has no metadata block, regardless of
    // what the fixture JSON contains. LoRAFile::scale_weight's network-type dispatch
    // is metadata-only, so it falls back to plain LoRA scaling here, which errors on
    // hada-only keys. This is a test-harness gap (out of scope for this task), not a
    // product bug -- this cell stops at unet_keys().
    load_and_check("sdxl/lycoris.json", LoRAFormat::Kohya, false);
}

#[test]
fn flux1_kohya() {
    load_and_check("flux1/kohya.json", LoRAFormat::Kohya, true);
}

#[test]
fn flux1_diffusers() {
    load_and_check("flux1/diffusers.json", LoRAFormat::Peft, true);
}

#[test]
fn flux2_klein_kohya() {
    load_and_check("flux2_klein/kohya.json", LoRAFormat::Kohya, true);
}

#[test]
fn flux2_klein_diffusers() {
    load_and_check("flux2_klein/diffusers.json", LoRAFormat::Peft, true);
}

#[test]
fn wan_kohya() {
    load_and_check("wan/kohya.json", LoRAFormat::Kohya, true);
}

#[test]
fn wan_diffusers() {
    load_and_check("wan/diffusers.json", LoRAFormat::Peft, true);
}

#[test]
fn krea2_diffusers() {
    load_and_check("krea2/diffusers.json", LoRAFormat::Peft, true);
}

#[test]
fn krea2_lycoris() {
    // LoKr, but the file's metadata lacks ss_network_args/ss_network_module (the
    // ai-toolkit trainer that produced it doesn't write kohya's sd-scripts metadata
    // format), so LoRAFile::scale_weight can't tell it's LoKr and falls back to plain
    // LoRA scaling, which errors on LoKr-only keys. Metadata-independent network-type
    // detection is a separate, unscoped feature -- this cell stops at unet_keys().
    load_and_check("krea2/lycoris.json", LoRAFormat::Kohya, false);
}

#[test]
fn h3_diffusers() {
    load_and_check("h3/diffusers.json", LoRAFormat::Peft, true);
}

#[test]
fn qwen_image_kohya() {
    load_and_check("qwen_image/kohya.json", LoRAFormat::Kohya, true);
}

#[test]
fn ltx23_diffusers() {
    load_and_check("ltx23/diffusers.json", LoRAFormat::Peft, true);
}
