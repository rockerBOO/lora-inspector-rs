import React from "react";

export function StatisticRow({
	baseName,
	l1Norm,
	l2Norm,
	matrixNorm,
	min,
	max,
	median,
	stdDev,
}) {
	return (
		<tr>
			<td>{baseName}</td>
			<td>{l1Norm?.toPrecision(4)}</td>
			<td>{l2Norm?.toPrecision(4)}</td>
			<td>{matrixNorm?.toPrecision(4)}</td>
			<td>{min?.toPrecision(4)}</td>
			<td>{max?.toPrecision(4)}</td>
			<td>{median?.toPrecision(4)}</td>
			<td>{stdDev?.toPrecision(4)}</td>
		</tr>
	);
}
