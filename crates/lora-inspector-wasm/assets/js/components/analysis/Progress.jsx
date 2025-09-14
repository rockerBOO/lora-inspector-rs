export function Progress({
	totalCount,
	currentCount,
	statisticProgress,
	startTime,
	currentItemName,
}) {
	const elapsedTime = performance.now() - startTime;
	const remaining =
		(elapsedTime * totalCount) / statisticProgress - elapsedTime * totalCount ||
		0;
	const perSecond = currentCount / (elapsedTime / 1_000);

	return (
		<div className="block-weights-container">
			<span>
				Loading statistics... {(statisticProgress * 100).toFixed(2)}%{" "}
				{currentCount}/{totalCount} {perSecond.toFixed(2)}it/s{" "}
				{(remaining / 1_000_000).toFixed(2)}s remaining {currentItemName}{" "}
			</span>
		</div>
	);
}
