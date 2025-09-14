import { useState } from "react";

export function TagFrequency({ tagFrequency }) {
	const [showMore, setShowMore] = useState(false);

	const allTags = Object.entries(tagFrequency).sort((a, b) => a[1] < b[1]);
	const sortedTags = showMore === false ? allTags.slice(0, 50) : allTags;

	return [
		...sortedTags.map(([tag, count], i) => {
			const alt = i % 2 > 0 ? " alt-row" : "";
			return (
				<div className={`tag-frequency${alt}`} key={tag}>
					<div>{count}</div>
					<div>{tag}</div>
				</div>
			);
		}),
		<div key="show-more">
			{showMore === false && allTags.length > sortedTags.length ? (
				<button
					type="button"
					onClick={() => {
						setShowMore(true);
					}}
				>
					Show more
				</button>
			) : (
				showMore === true && (
					<button
						type="button"
						onClick={() => {
							setShowMore(false);
						}}
					>
						Show less
					</button>
				)
			)}
		</div>,
	];
}
