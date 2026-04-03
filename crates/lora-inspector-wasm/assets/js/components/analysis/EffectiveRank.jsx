import { useEffect, useState } from "react";
import { trySyncMessage } from "../../message.js";
import { Progress } from "./Progress.jsx";

const HEALTH_LABELS = {
	Good: "GOOD",
	Ok: "OK",
	Weak: "WEAK",
	Poor: "POOR",
	Collapsed: "COLLAPSED",
};

function healthClass(health) {
	if (!health) return "";
	return `rank-health-${health.toLowerCase()}`;
}

function median(values) {
	if (values.length === 0) return 0;
	const sorted = [...values].sort((a, b) => a - b);
	const mid = Math.floor(sorted.length / 2);
	return sorted.length % 2 === 0
		? (sorted[mid - 1] + sorted[mid]) / 2
		: sorted[mid];
}

function EffectiveRankRow({ scale, metrics }) {
	const health = metrics?.health;
	return (
		<tr className={healthClass(health)}>
			<td>{scale.base_name}</td>
			<td className="number">{scale.eff_scale.toPrecision(4)}</td>
			<td>{scale.is_outlier ? "⚠" : ""}</td>
			<td className="number">{metrics?.nominal_rank ?? "—"}</td>
			<td className="number">
				{metrics?.effective_rank?.toPrecision(3) ?? "—"}
			</td>
			<td>{metrics ? (HEALTH_LABELS[health] ?? health) : "—"}</td>
			<td className="number">{metrics?.balance?.toPrecision(3) ?? "—"}</td>
			<td className="number">{metrics?.top1_energy?.toPrecision(3) ?? "—"}</td>
			<td className="number">
				{metrics?.dominance != null ? metrics.dominance.toPrecision(3) : "—"}
			</td>
		</tr>
	);
}

export function EffectiveRank({ filename, worker }) {
	const [triggered, setTriggered] = useState(false);
	const [scales, setScales] = useState(null);
	const [metricsMap, setMetricsMap] = useState({});
	const [currentCount, setCurrentCount] = useState(0);
	const [totalCount, setTotalCount] = useState(0);
	const [progress, setProgress] = useState(0);
	const [startTime, setStartTime] = useState(undefined);
	const [currentBaseName, setCurrentBaseName] = useState("");
	const [phase, setPhase] = useState("idle"); // idle | scales | metrics | done
	const [effectiveRankFilename, setEffectiveRankFilename] = useState("");

	// Reset when file changes
	useEffect(() => {
		if (effectiveRankFilename !== filename) {
			setTriggered(false);
			setScales(null);
			setMetricsMap({});
			setCurrentCount(0);
			setTotalCount(0);
			setProgress(0);
			setStartTime(undefined);
			setPhase("idle");
			setEffectiveRankFilename(filename);
		}
	}, [effectiveRankFilename, filename]);

	// Phase 1: per-layer effective_scale with progress, then compute outliers in JS
	useEffect(() => {
		if (!triggered) return;

		const fetchScales = async (baseNames) => {
			console.time("[EffectiveRank] phase1: effective_scale");
			setPhase("scales");
			setTotalCount(baseNames.length);
			setStartTime(performance.now());

			let count = 0;
			const results = await Promise.allSettled(
				baseNames.map(async (baseName) => {
					const resp = await trySyncMessage(
						{
							messageType: "effective_scale",
							name: filename,
							baseName,
						},
						worker,
						["messageType", "baseName"],
					);
					count += 1;
					setCurrentBaseName(baseName);
					setCurrentCount(count);
					setProgress(count / baseNames.length);
					return { base_name: baseName, eff_scale: resp.effScale ?? 0 };
				}),
			);
			console.timeEnd("[EffectiveRank] phase1: effective_scale");

			const collected = results
				.filter((r) => r.status === "fulfilled" && r.value.eff_scale > 0)
				.map((r) => r.value);

			const med = median(collected.map((s) => s.eff_scale));
			const threshold = Math.max(med, 1e-10) * 1.5;
			const withOutliers = collected.map((s) => ({
				...s,
				is_outlier: s.eff_scale > threshold,
			}));
			withOutliers.sort((a, b) => a.base_name.localeCompare(b.base_name));
			return withOutliers;
		};

		trySyncMessage(
			{ messageType: "base_names", name: filename, reply: true },
			worker,
		).then((resp) => fetchScales(resp.baseNames).then(setScales));
	}, [triggered, filename, worker]);

	// Phase 2: per-layer rank_metrics with progress
	useEffect(() => {
		if (!scales) return;

		const fetchMetrics = async () => {
			console.time("[EffectiveRank] phase2: rank_metrics");
			setPhase("metrics");
			setCurrentCount(0);
			setTotalCount(scales.length);
			setProgress(0);
			setStartTime(performance.now());

			let count = 0;
			await Promise.allSettled(
				scales.map(async (scale) => {
					const resp = await trySyncMessage(
						{
							messageType: "rank_metrics",
							name: filename,
							baseName: scale.base_name,
						},
						worker,
						["messageType", "baseName"],
					);
					count += 1;
					setCurrentBaseName(resp.baseName);
					setCurrentCount(count);
					setProgress(count / scales.length);
					if (resp.metrics) {
						setMetricsMap((prev) => ({
							...prev,
							[resp.baseName]: resp.metrics,
						}));
					}
				}),
			);
			console.timeEnd("[EffectiveRank] phase2: rank_metrics");
			console.log(
				`[EffectiveRank] completed ${scales.length} layers in ${filename}`,
			);
			setPhase("done");
		};

		fetchMetrics();
	}, [scales, filename, worker]);

	if (!triggered) {
		return (
			<div className="effective-rank-container">
				<button
					type="button"
					className="primary"
					onClick={(e) => {
						e.preventDefault();
						setTriggered(true);
					}}
				>
					Calculate effective rank
				</button>
			</div>
		);
	}

	if (phase === "scales" || (phase === "metrics" && !scales)) {
		return (
			<div className="effective-rank-container">
				<Progress
					totalCount={totalCount}
					currentCount={currentCount}
					statisticProgress={progress}
					startTime={startTime}
					currentItemName={currentBaseName}
				/>
			</div>
		);
	}

	const outlierCount = scales?.filter((s) => s.is_outlier).length ?? 0;

	return (
		<div className="effective-rank-container">
			{phase === "metrics" && (
				<Progress
					totalCount={totalCount}
					currentCount={currentCount}
					statisticProgress={progress}
					startTime={startTime}
					currentItemName={currentBaseName}
				/>
			)}

			{outlierCount > 0 && (
				<p className="rank-outlier-summary">
					⚠ {outlierCount} outlier{outlierCount > 1 ? "s" : ""} (eff_scale &gt;
					1.5× median)
				</p>
			)}

			<table>
				<thead>
					<tr>
						<th title="Layer key prefix shared by lora_up, lora_down, and alpha tensors">
							base name
						</th>
						<th title="norm(up @ down) × (alpha / rank) — the actual weight delta this layer applies at strength 1.0">
							eff scale
						</th>
						<th title="eff scale is more than 1.5× the median across all layers">
							outlier
						</th>
						<th title="Declared bottleneck size of the LoRA layer">rank</th>
						<th title="exp(entropy(s² / Σs²)) — how many singular value dimensions are actually used. Equal to rank when perfectly balanced, approaches 1 when collapsed">
							eff rank
						</th>
						<th title="GOOD ≥ 0.75 · rank | OK ≥ 0.50 | WEAK ≥ 0.25 | POOR &lt; 0.25 | COLLAPSED ≈ rank-1">
							health
						</th>
						<th title="eff rank / rank — 1.0 means all dimensions contribute equally; low values mean rank is being wasted">
							balance
						</th>
						<th title="s[0]² / Σs² — fraction of total energy in the single strongest direction. 1.0 means the layer has effectively collapsed to rank-1">
							top1 energy
						</th>
						<th title="s[0] / s[1] — ratio of the two largest singular values. High values mean one direction strongly dominates the rest">
							dominance
						</th>
					</tr>
				</thead>
				<tbody>
					{scales?.map((scale) => (
						<EffectiveRankRow
							key={scale.base_name}
							scale={scale}
							metrics={metricsMap[scale.base_name]}
						/>
					))}
				</tbody>
			</table>

			<dl className="rank-legend">
				<dt>eff scale</dt>
				<dd>
					norm(up @ down) × (alpha / rank) — the actual weight delta this layer
					applies at strength 1.0
				</dd>
				<dt>outlier</dt>
				<dd>eff scale is more than 1.5× the median across all layers</dd>
				<dt>rank</dt>
				<dd>Declared bottleneck size of the LoRA layer</dd>
				<dt>eff rank</dt>
				<dd>
					exp(entropy(s² / Σs²)) — how many singular value dimensions are
					actually used. Equal to rank when perfectly balanced, approaches 1
					when collapsed
				</dd>
				<dt>health</dt>
				<dd>
					GOOD ≥ 0.75 · rank | OK ≥ 0.50 | WEAK ≥ 0.25 | POOR &lt; 0.25 |
					COLLAPSED ≈ rank-1
				</dd>
				<dt>balance</dt>
				<dd>
					eff rank / rank — 1.0 means all dimensions contribute equally; low
					values mean rank is being wasted
				</dd>
				<dt>top1 energy</dt>
				<dd>
					s[0]² / Σs² — fraction of total energy in the single strongest
					direction. 1.0 means the layer has effectively collapsed to rank-1
				</dd>
				<dt>dominance</dt>
				<dd>
					s[0] / s[1] — ratio of the two largest singular values. High values
					mean one direction strongly dominates the rest
				</dd>
			</dl>
		</div>
	);
}
