/**
 * Ground truth metadata fixture from boo.safetensors.
 * Used across component tests to ensure consistent, realistic test data.
 *
 * Keys absent from this Map represent fields not present in the file —
 * components should display "—" for those.
 */
export const booMetadata = new Map([
	["ss_training_started_at", "1676884936"],
	[
		"ss_training_comment",
		"dimension is resized from 128 to 4; Trained by: rockerBOO. Identifier: cp-ede Class: anime Learning Rate: 1.0, Batch Size: 1, Gradient Accumulation Steps: 1",
	],
	["ss_sd_model_name", "v1-5-pruned-emaonly.ckpt"],
	[
		"ss_new_sd_model_hash",
		"74b62bfc40e092a456de76ff3280ea6eb8e88d542d9cf7977eedc59731fcfe16",
	],
	["ss_sd_model_hash", "9e689848"],
	["ss_session_id", "682850588"],
	[
		"ss_sd_scripts_commit_hash",
		"b612d0b091213f39f4864b4cfe63a44f1e1974d7",
	],
	["ss_network_module", "kohya-ss/lora"],
	["ss_network_dim", "4"],
	["ss_network_alpha", "4.0"],
	["ss_network_args", "null"],
	["ss_lr_scheduler", "cosine"],
	["ss_lr_warmup_steps", "10"],
	["ss_unet_lr", "1.0"],
	["ss_text_encoder_lr", "0.5"],
	["ss_optimizer", "dadaptation.dadapt_adam.DAdaptAdam"],
	["ss_seed", "42"],
	["ss_clip_skip", "1"],
	["ss_mixed_precision", "fp16"],
	["ss_full_fp16", "False"],
	["ss_epoch", "1"],
	["ss_max_train_steps", "20000"],
	["ss_num_train_images_per_concept", "30300"],
	["ss_num_batches_per_epoch", "60600"],
	["ss_batch_size_per_gpu", "1"],
	["ss_gradient_accumulation_steps", "1"],
	["ss_noise_offset", "-0.1"],
	["ss_debiased_estimation", "False"],
	["ss_max_token_length", "None"],
	["ss_gradient_checkpointing", "False"],
]);

/**
 * Minimal metadata with only the most basic fields.
 * Useful for testing absent-field (em-dash) behavior.
 */
export const minimalMetadata = new Map([
	["ss_network_module", "kohya-ss/lora"],
	["ss_network_dim", "4"],
	["ss_network_alpha", "4.0"],
]);
