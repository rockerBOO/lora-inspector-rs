export function Progress({
	totalCount,
	currentCount,
	statisticProgress,
	startTime,
	currentItemName,
}) {
	const elapsedMs = startTime != null ? performance.now() - startTime : 0;
	const msPerItem = currentCount > 0 ? elapsedMs / currentCount : 0;
	const remainingMs = msPerItem * (totalCount - currentCount);
	const perSecond = elapsedMs > 0 ? currentCount / (elapsedMs / 1_000) : 0;

	return (
		<div className="block-weights-container">
			<span>
				Loading... {(statisticProgress * 100).toFixed(1)}% {currentCount}/
				{totalCount} {perSecond.toFixed(2)}it/s{" "}
				{(remainingMs / 1_000).toFixed(1)}s remaining {currentItemName}
			</span>
		</div>
	);
}
