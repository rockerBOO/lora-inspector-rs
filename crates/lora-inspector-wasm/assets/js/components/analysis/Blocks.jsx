import React, { useState, useEffect, useRef } from "react";
import { MetaAttribute } from "../ui/MetaAttribute.jsx";
import { listenProgress, trySyncMessage } from "../../message.js";

export function Blocks({ filename, worker }) {
	const [hasBlockWeights, setHasBlockWeights] = useState(false);
	const [magBlocks, setMagBlocks] = useState({});
	const [normProgress, setNormProgress] = useState(0);
	const [currentCount, setCurrentCount] = useState(0);
	const [totalCount, setTotalCount] = useState(0);
	const [blockFilename, setBlockFilename] = useState("");
	const [startTime, setStartTime] = useState(undefined);
	const [currentBaseName, setCurrentBaseName] = useState("");
	const [canHaveBlockWeights, setCanHaveBlockWeights] = useState(false);

	const chartRefs = useRef(
		Array.from(Array(4).keys()).map(() => React.createRef()),
	);

	// Reset
	useEffect(() => {
		if (blockFilename !== filename) {
			setHasBlockWeights(false);
			setMagBlocks({});
			setCurrentCount(0);
			setNormProgress(0);
			setTotalCount(0);
			setStartTime(undefined);
			setCanHaveBlockWeights(false);
			setBlockFilename(filename);
		}
	}, [blockFilename, filename]);

	useEffect(() => {
		if (!hasBlockWeights) {
			return;
		}

		setStartTime(performance.now());

		listenProgress("l2_norms_progress", filename).then(async (getProgress) => {
			let progress = await getProgress().next();
			while (progress) {
				const value = progress.value;

				if (!value) {
					break;
				}

				setCurrentBaseName(value.baseName);
				setCurrentCount(value.currentCount);
				setTotalCount(value.totalCount);
				setNormProgress(value.currentCount / value.totalCount);
				progress = await getProgress().next();
			}
		});

		trySyncMessage(
			{
				messageType: "l2_norm",
				name: filename,
				reply: true,
			},
			worker,
		).then((resp) => {
			setMagBlocks(resp.norms);
		});

		return function cleanup() {};
	}, [hasBlockWeights, filename, worker]);

	useEffect(() => {
		trySyncMessage(
			{
				messageType: "network_type",
				name: filename,
				reply: true,
			},
			worker,
		).then((resp) => {
			if (
				resp.networkType === "LoRA" ||
				resp.networkType === "LoRAFA" ||
				resp.networkType === "DyLoRA" ||
				resp.networkType === "GLoRA" ||
				resp.networkType === "LoHA" ||
				resp.networkType === "LoKr" ||
				resp.networkType === "DiagOFT" ||
				resp.networkType === "BOFT" ||
				// Assuming networkType of none could have block weights
				resp.networkType === undefined
			) {
				setCanHaveBlockWeights(true);
			}
		});
	}, [filename, worker]);

	useEffect(() => {
		if (!chartRefs.current[0]) {
			return;
		}

		const makeChart = (dataset, chartRef) => {
			const data = {
				// A labels array that can contain any sort of values
				labels: dataset.map(([k, _]) => k),
				// Our series array that contains series objects or in this case series data arrays
				series: [
					dataset.map(([_k, v]) => v.mean),
					// dataset.map(([k, v]) => strBlocks.get(k)),
				],
			};
			const chart = new Chartist.Line(chartRef.current, data, {
				chartPadding: {
					right: 60,
					top: 30,
					bottom: 30,
				},
				// seriesBarDistance: 15,
				fullWidth: true,
				axisX: {
					// showGrid: false,
					// offset: 10,
					// offset: -60,
					// position: "start",
				},
				axisY: {
					offset: 60,
					// scaleMinSpace: 100,
					// position: "end",
				},
				plugins: [
					Chartist.plugins.ctPointLabels({
						labelOffset: {
							x: 10,
							y: -10,
						},
						textAnchor: "middle",
						labelInterpolationFnc: (value) => value.toPrecision(4),
					}),
				],
			});

			let seq = 0;

			// Once the chart is fully created we reset the sequence
			chart.on("created", () => {
				seq = 0;
			});

			chart.on("draw", (data) => {
				if (data.type === "point") {
					// If the drawn element is a line we do a simple opacity fade in. This could also be achieved using CSS3 animations.
					data.element.animate({
						opacity: {
							// The delay when we like to start the animation
							begin: seq++ * 40,
							// Duration of the animation
							dur: 90,
							// The value where the animation should start
							from: 0,
							// The value where it should end
							to: 1,
						},
						x1: {
							begin: seq++ * 20,
							dur: 90,
							from: data.x - 20,
							to: data.x,
							// You can specify an easing function name or use easing functions from Chartist.Svg.Easing directly
							easing: Chartist.Svg.Easing.easeOutQuart,
						},
					});
				}
			});
		};

		if (Object.keys(magBlocks).length > 0) {
			Object.keys(magBlocks).forEach((k, i) => {
				if (magBlocks[k].size === 0) {
					return;
				}

				makeChart(
					// We are removing elements that are 0 because they cause the chart to find them as undefined
					Array.from(magBlocks[k]).filter(([_k, v]) => v.mean !== 0),
					chartRefs.current[i],
				);
			});
		}
	}, [magBlocks]);

	if (!canHaveBlockWeights) {
		return (
			<div className="block-weights-container">
				Block weights not supported for this network type or precision.
			</div>
		);
	}

	if (!hasBlockWeights) {
		return (
			<div className="block-weights-container">
				<button
					type="button"
					className="primary"
					onClick={(e) => {
						e.preventDefault();
						setHasBlockWeights((state) => !state);
					}}
				>
					Get block weights
				</button>
			</div>
		);
	}

	let magBlockWeights = [];
	if (Object.keys(magBlocks).length > 0) {
		magBlockWeights = Object.entries(magBlocks).map(([magKey, mags], i) => {
			if (mags.size === 0) {
				return undefined;
			}

			return (
				<div key={`mag-block-${magKey}`}>
					<h3>{magKey} block weights</h3>
					<div ref={chartRefs.current[i]} className="chart" />
					<div className="block-weights">
						{Array.from(mags).map(([k, v]) => {
							return (
								<div key={k}>
									<MetaAttribute
										className="unet-block"
										name={`${k} avg l2 norm  ${v.metadata.type}`}
										value={v.mean.toPrecision(6)}
										valueClassName="number"
									/>
								</div>
							);
						})}
					</div>
				</div>
			);
		});
	}

	if (magBlockWeights.length === 0 && hasBlockWeights === true) {
		const elapsedTime = performance.now() - startTime;
		const remaining =
			(elapsedTime * totalCount) / normProgress - elapsedTime * totalCount;
		const perSecond = currentCount / (elapsedTime / 1_000);

		return (
			<div className="block-weights-container">
				<span>
					Loading block weights... {(normProgress * 100).toFixed(2)}%{" "}
					{currentCount}/{totalCount} {perSecond.toFixed(2)}it/s{" "}
					{(remaining / 1_000_000).toFixed(2)}s remaining {currentBaseName}
				</span>
			</div>
		);
	}

	return <div className="block-weights-container">{magBlockWeights}</div>;
}
