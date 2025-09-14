export function WaveletLossWeights({ weights }) {
	return (
		<div>
			{Object.entries(weights).map(([k, v]) => (
				<div className="weightedLoss" key={k}>
					<h4>{k}</h4>
					<div>{v}</div>
				</div>
			))}
		</div>
	);
}
