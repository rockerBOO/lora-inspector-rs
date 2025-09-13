import { Advanced } from "../analysis/Advanced.jsx";
import { Precision } from "../analysis/Precision.jsx";
import { Weight } from "../analysis/Weight.jsx";
import { CaptionDropout } from "../dataset/CaptionDropout.jsx";
import { Dataset } from "../dataset/Dataset.jsx";
import { PretrainedModel } from "../metadata/PretrainedModel.jsx";
import { LoRANetwork } from "../network/LoRANetwork.jsx";
import { Network } from "../network/Network.jsx";
import { Batch } from "../training/Batch.jsx";
import { EpochStep } from "../training/EpochStep.jsx";
import { LRScheduler } from "../training/LRScheduler.jsx";
import { Loss } from "../training/Loss.jsx";
import { Noise } from "../training/Noise.jsx";
import { Optimizer } from "../training/Optimizer.jsx";
import { WaveletLoss } from "../training/WaveletLoss.jsx";
import { Headline } from "./Headline.jsx";

export function Main({ metadata, filename, worker }) {
	if (!metadata) {
		return (
			<main>
				<div>No metadata for this file</div>
				<Headline filename={filename} />
				<div className="row space-apart">
					<LoRANetwork
						metadata={metadata}
						filename={filename}
						worker={worker}
					/>
					<Precision filename={filename} worker={worker} />
				</div>
				<Weight metadata={metadata} filename={filename} worker={worker} />
				<Advanced metadata={metadata} filename={filename} worker={worker} />
			</main>
		);
	}

	return (
		<main>
			<PretrainedModel metadata={metadata} />
			<Network metadata={metadata} filename={filename} worker={worker} />
			<LRScheduler metadata={metadata} />
			<Optimizer metadata={metadata} />
			<Weight metadata={metadata} filename={filename} worker={worker} />
			<EpochStep metadata={metadata} />
			<Batch metadata={metadata} />
			<Noise metadata={metadata} />
			<Loss metadata={metadata} />
			<WaveletLoss metadata={metadata} />
			<CaptionDropout metadata={metadata} />
			<Dataset metadata={metadata} />
			<Advanced metadata={metadata} filename={filename} worker={worker} />
		</main>
	);
}
