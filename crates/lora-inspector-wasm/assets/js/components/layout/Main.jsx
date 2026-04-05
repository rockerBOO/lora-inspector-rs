import { Advanced } from "../analysis/Advanced.jsx";
import { Blocks } from "../analysis/Blocks.jsx";
import { Precision } from "../analysis/Precision.jsx";
import { Dataset } from "../dataset/Dataset.jsx";
import { ModelSpec } from "../metadata/ModelSpec.jsx";
import { PretrainedModel } from "../metadata/PretrainedModel.jsx";
import { LoRANetwork } from "../network/LoRANetwork.jsx";
import { Network } from "../network/Network.jsx";
import { Batch } from "../training/Batch.jsx";
import { EpochStep } from "../training/EpochStep.jsx";
import { Experimental } from "../training/Experimental.jsx";
import { FluxTraining } from "../training/FluxTraining.jsx";
import { Loss } from "../training/Loss.jsx";
import { LRScheduler } from "../training/LRScheduler.jsx";
import { Optimizer } from "../training/Optimizer.jsx";
import { WaveletLoss } from "../training/WaveletLoss.jsx";
import { MetaAttribute } from "../ui/MetaAttribute.jsx";
import { Headline } from "./Headline.jsx";
import { Section } from "./Section.jsx";

export function Main({ metadata, filename, worker }) {
	if (!metadata) {
		return (
			<main>
				<Headline filename={filename} />
				<div className="row space-apart">
					<LoRANetwork
						metadata={metadata}
						filename={filename}
						worker={worker}
					/>
					<Precision filename={filename} worker={worker} />
				</div>
				<Blocks filename={filename} worker={worker} />
				<Advanced metadata={metadata} filename={filename} worker={worker} />
			</main>
		);
	}

	return (
		<main>
			<Section id="metadata" label="Metadata">
				<PretrainedModel metadata={metadata} />
				<ModelSpec metadata={metadata} />
			</Section>

			<Section id="network" label="Network">
				<Network metadata={metadata} filename={filename} worker={worker} />
				{metadata.has("ss_scale_weight_norms") && (
					<div className="row space-apart">
						<MetaAttribute
							name="Scale Weight Norms"
							valueClassName="number"
							value={metadata.get("ss_scale_weight_norms")}
						/>
					</div>
				)}
			</Section>

			<Section id="training" label="Training">
				<LRScheduler metadata={metadata} />
				<EpochStep metadata={metadata} />
				<Batch metadata={metadata} />
				<FluxTraining metadata={metadata} />
				<Loss metadata={metadata} />
				<WaveletLoss metadata={metadata} />
				<div className="row space-apart">
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
					{metadata.has("ss_gradient_checkpointing_cpu_offload") && (
						<MetaAttribute
							name="Gradient checkpointing CPU offload"
							valueClassName="number"
							value={metadata.get("ss_gradient_checkpointing_cpu_offload")}
						/>
					)}
				</div>
				<Experimental metadata={metadata} />
			</Section>

			<Section id="optimizer" label="Optimizer">
				<Optimizer metadata={metadata} />
				<div className="row space-apart">
					<MetaAttribute
						name="Max Grad Norm"
						valueClassName="number"
						value={metadata.get("ss_max_grad_norm")}
					/>
					<MetaAttribute
						name="CLIP Skip"
						valueClassName="number"
						value={metadata.get("ss_clip_skip")}
					/>
				</div>
			</Section>

			<Section id="dataset" label="Dataset">
				<Dataset metadata={metadata} />
			</Section>

			<Section id="advanced" label="Advanced">
				<Blocks filename={filename} worker={worker} />
				<Advanced metadata={metadata} filename={filename} worker={worker} />
			</Section>
		</main>
	);
}
