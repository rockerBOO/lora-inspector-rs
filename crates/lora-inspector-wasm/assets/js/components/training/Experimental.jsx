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

export function Experimental({ metadata }) {
	const selfFlowKeys = SELF_FLOW_KEYS.filter(([k]) => metadata.has(k));
	const cdcFmKeys = CDC_FM_KEYS.filter(([k]) => metadata.has(k));
	if (selfFlowKeys.length === 0 && cdcFmKeys.length === 0) return null;

	return (
		<details className="experimental" open>
			<summary>Experimental</summary>
			{selfFlowKeys.length > 0 && (
				<div className="experimental-group">
					<h3>Self Flow</h3>
					<div className="row space-apart">
						{selfFlowKeys.map(([k, label]) => (
							<MetaAttribute key={k} name={label} value={metadata.get(k)} />
						))}
					</div>
				</div>
			)}
			{cdcFmKeys.length > 0 && (
				<div className="experimental-group">
					<h3>CDC-FM</h3>
					<div className="row space-apart">
						{cdcFmKeys.map(([k, label]) => (
							<MetaAttribute key={k} name={label} value={metadata.get(k)} />
						))}
					</div>
				</div>
			)}
		</details>
	);
}
