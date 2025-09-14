import { useEffect, useState } from "react";
import { listenProgress, trySyncMessage } from "../../message.js";
import { Progress } from "./Progress.jsx";
import { StatisticRow } from "./StatisticRow.jsx";

const DEBUG = new URLSearchParams(document.location.search).has("DEBUG");

export function Statistics({ baseNames, filename, worker }) {
	const [calcStatistics, setCalcStatistics] = useState(false);
	const [hasStatistics, setHasStatistics] = useState(false);
	const [bases, setBases] = useState([]);
	const [statisticProgress, setStatisticProgress] = useState(0);
	const [currentCount, setCurrentCount] = useState(0);
	const [totalCount, setTotalCount] = useState(0);
	const [startTime, setStartTime] = useState(undefined);
	const [currentBaseName, setCurrentBaseName] = useState("");

	useEffect(() => {
		if (!calcStatistics) {
			return;
		}
		if (baseNames.length === 0) {
			return;
		}

		console.log("Calculating statistics...");
		console.time("get statistics");
		setStartTime(performance.now());

		let progress = 0;
		Promise.allSettled(
			baseNames.map(async (baseName) => {
				return trySyncMessage(
					{ messageType: "norms", name: filename, baseName },
					worker,
					["messageType", "baseName"],
				).then((resp) => {
					progress += 1;
					setCurrentBaseName(resp.baseName);
					setCurrentCount(progress);
					setTotalCount(baseNames.length);
					setStatisticProgress(progress / baseNames.length);
					bases.push({ baseName: resp.baseName, stat: resp.norms });
					setBases(bases);
					return { baseName: resp.baseName, stat: resp.norms };
				});
			}),
		).then((results) => {
			progress = 0;
			const bases = results
				.filter((v) => v.status === "fulfilled")
				.map((v) => v.value);
			setBases(bases);
			setHasStatistics(true);
			console.timeEnd("get statistics");
		});
	}, [filename, calcStatistics, bases, baseNames, worker]);

	useEffect(() => {
		if (!calcStatistics) {
			return;
		}
		if (baseNames.length === 0) {
			return;
		}

		setStartTime(performance.now());
		listenProgress("scale_weight_progress", filename).then(
			async (getProgress) => {
				let progress;
				while (await newFunction()) {
					const value = progress.value;
					if (!value) {
						break;
					}
					setCurrentBaseName(value.baseName);
					// Note: These setter functions are referenced but not defined in original
					// Keeping the logic structure but commenting out undefined functions
					// setCurrentScaleWeightCount(value.currentCount);
					// setTotalScaleWeightCount(value.totalCount);
					// setScaleWeightProgress(value.currentCount / value.totalCount);
				}
				async function newFunction() {
					progress = await getProgress().next();
					return progress;
				}
			},
		);

		return function cleanup() {};
	}, [calcStatistics, filename, baseNames]);

	if (!hasStatistics && !calcStatistics) {
		return (
			<div>
				<button
					type="button"
					onClick={(e) => {
						e.preventDefault();
						setCalcStatistics(true);
					}}
				>
					Calculate statistics
				</button>
			</div>
		);
	}

	// const teLayers = compileTextEncoderLayers(bases);
	// const unetLayers = compileUnetLayers(bases);

	return (
		<>
			{DEBUG && (
				<div
					style={{
						display: "grid",
						justifyContent: "flex-end",
					}}
				>
					<button
						type="button"
						onClick={() => {
							// console.log("teLayers", teLayers);
							// console.log("unetLayers", unetLayers);
							console.log(
								"bases",
								bases.map((v) => ({
									...v,
									stat: Object.fromEntries(v.stat),
								})),
							);
						}}
					>
						debug stats
					</button>
				</div>
			)}

			{calcStatistics && !hasStatistics && (
				<Progress
					totalCount={totalCount}
					currentCount={currentCount}
					statisticProgress={statisticProgress}
					startTime={startTime}
					currentItemName={currentBaseName}
				/>
			)}

			<table>
				<thead>
					<tr>
						<th>base name</th>
						<th>l1 norm</th>
						<th>l2 norm</th>
						<th>matrix norm</th>
						<th>min</th>
						<th>max</th>
						<th>median</th>
						<th>std_dev</th>
					</tr>
				</thead>
				<tbody>
					{bases.map((base, i) => (
						<StatisticRow
							key={`base-${base.baseName || i}`}
							baseName={base.baseName}
							l1Norm={base.stat?.get("l1_norm")}
							l2Norm={base.stat?.get("l2_norm")}
							matrixNorm={base.stat?.get("matrix_norm")}
							min={base.stat?.get("min")}
							max={base.stat?.get("max")}
							median={base.stat?.get("median")}
							stdDev={base.stat?.get("std_dev")}
						/>
					))}
				</tbody>
			</table>

			{/* 
			{teLayers.length > 0 && (
				<>
					<div><h2>Text Encoder Architecture</h2></div>
					<div id="te-architecture">
						<TEArchitecture layers={teLayers} />
					</div>
				</>
			)}

			<div><h2>UNet Architecture</h2></div>
			<div id="unet-architecture">
				<UNetArchitecture layers={unetLayers} />
			</div>
			*/}
		</>
	);
}
