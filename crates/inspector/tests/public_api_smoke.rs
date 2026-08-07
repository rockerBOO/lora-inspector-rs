#[test]
fn lora_format_is_publicly_accessible() {
    assert_ne!(inspector::LoRAFormat::Kohya, inspector::LoRAFormat::Peft);
}
