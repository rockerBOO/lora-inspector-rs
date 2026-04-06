import { MetaAttribute } from "../ui/MetaAttribute.jsx";

const CDC_FM_KEYS = [
	["ss_use_cdc_fm", "CDC-FM enabled"],
	["ss_cdc_k_neighbors", "K neighbors"],
	["ss_cdc_k_bandwidth", "K bandwidth"],
	["ss_cdc_d_cdc", "CDC dimension"],
	["ss_cdc_gamma", "Gamma"],
	["ss_cdc_adaptive_k", "Adaptive K"],
	["ss_cdc_min_bucket_size", "Min bucket size"],
];

const SELF_FLOW_KEYS = [
	["ss_self_flow", "Self flow"],
	["ss_self_flow_ema_decay", "EMA decay"],
	["ss_self_flow_gamma", "Gamma"],
	["ss_self_flow_gamma_warmup_steps", "Gamma warmup steps"],
	["ss_self_flow_mask_ratio", "Mask ratio"],
	["ss_self_flow_student_layer", "Student layer"],
	["ss_self_flow_teacher_layer", "Teacher layer"],
	["ss_self_flow_teacher_coupling_decay", "Teacher coupling decay"],
	["ss_self_flow_teacher_coupling_prob", "Teacher coupling prob"],
	["ss_self_flow_teacher_mismatch_ratio", "Teacher mismatch ratio"],
];

const NOISE_OFFSET_KEYS = [
	["ss_noise_offset", "Offset"],
	["ss_noise_offset_random_strength", "Random strength"],
	["ss_adaptive_noise_scale", "Adaptive scale"],
];

const MULTIRES_NOISE_KEYS = [
	["ss_multires_noise_iterations", "Iterations"],
	["ss_multires_noise_discount", "Discount"],
	["ss_ip_noise_gamma", "IP noise gamma"],
	["ss_ip_noise_gamma_random_strength", "IP random strength"],
];

const SNR_KEYS = [
	["ss_debiased_estimation", "Debiased estimation"],
	["ss_min_snr_gamma", "Min SNR gamma"],
	["ss_zero_terminal_snr", "Zero terminal SNR"],
];

function ExperimentalGroup({ label, keys, metadata }) {
	const present = keys.filter(([k]) => metadata.has(k));
	if (present.length === 0) return null;
	return (
		<div className="experimental-group">
			<h3>{label}</h3>
			<div className="row space-apart">
				{present.map(([k, name]) => (
					<MetaAttribute key={k} name={name} value={metadata.get(k)} />
				))}
			</div>
		</div>
	);
}

export function Experimental({ metadata }) {
	const selfFlowKeys = SELF_FLOW_KEYS.filter(([k]) => metadata.has(k));
	const cdcFmKeys = CDC_FM_KEYS.filter(([k]) => metadata.has(k));
	const noiseOffsetKeys = NOISE_OFFSET_KEYS.filter(([k]) => metadata.has(k));
	const multiresKeys = MULTIRES_NOISE_KEYS.filter(([k]) => metadata.has(k));
	const snrKeys = SNR_KEYS.filter(([k]) => metadata.has(k));

	const hasAny =
		selfFlowKeys.length > 0 ||
		cdcFmKeys.length > 0 ||
		noiseOffsetKeys.length > 0 ||
		multiresKeys.length > 0 ||
		snrKeys.length > 0;

	if (!hasAny) return null;

	return (
		<details className="experimental" open>
			<summary>Experimental</summary>
			<ExperimentalGroup
				label="Self Flow"
				keys={SELF_FLOW_KEYS}
				metadata={metadata}
			/>
			<ExperimentalGroup
				label="CDC-FM"
				keys={CDC_FM_KEYS}
				metadata={metadata}
			/>
			<ExperimentalGroup
				label="Noise offset"
				keys={NOISE_OFFSET_KEYS}
				metadata={metadata}
			/>
			<ExperimentalGroup
				label="MultiRes noise"
				keys={MULTIRES_NOISE_KEYS}
				metadata={metadata}
			/>
			<ExperimentalGroup label="SNR" keys={SNR_KEYS} metadata={metadata} />
		</details>
	);
}
