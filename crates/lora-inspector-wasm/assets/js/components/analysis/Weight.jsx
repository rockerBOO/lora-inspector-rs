import React from "react";
import { MetaAttribute } from "../ui/MetaAttribute.jsx";
import { Precision } from "./Precision.jsx";
import { Blocks } from "./Blocks.jsx";

export function Weight({ metadata, filename, worker }) {
	if (!metadata) {
		return <Blocks filename={filename} worker={worker} />;
	}

	return [
		<div key="norms" className="row space-apart">
			<MetaAttribute
				name="Max Grad Norm"
				valueClassName="number"
				value={metadata.get("ss_max_grad_norm")}
			/>
			<MetaAttribute
				name="Scale Weight Norms"
				valueClassName="number"
				value={metadata.get("ss_scale_weight_norms")}
			/>
			<MetaAttribute
				name="CLIP Skip"
				valueClassName="number"
				value={metadata.get("ss_clip_skip")}
			/>
		</div>,
		<div key="precision" className="row space-apart">
			<Precision filename={filename} worker={worker} />
			<MetaAttribute
				name="Mixed precision"
				valueClassName="number"
				value={metadata.get("ss_mixed_precision")}
			/>
			{metadata.has("ss_full_fp16") && (
				<MetaAttribute
					name="Full fp16"
					valueClassName="number"
					value={metadata.get("ss_full_fp16")}
				/>
			)}
			{metadata.has("ss_full_bf16") && (
				<MetaAttribute
					name="Full bf16"
					valueClassName="number"
					value={metadata.get("ss_full_bf16")}
				/>
			)}
			<MetaAttribute
				name="fp8 base"
				valueClassName="number"
				value={metadata.get("ss_fp8_base")}
			/>
		</div>,
		<div key="blocks">
			<Blocks filename={filename} worker={worker} />
		</div>,
	];
}
