import { WeightIn } from "../ui/SVGComponents.jsx";

export function Sampler({ conv }) {
	return (
		<svg
			className="sampler-layer"
			width="4em"
			height="85"
			role="img"
			aria-label="Sampler layer architecture diagram"
		>
			<title>Sampler Layer</title>
			<defs>
				<marker
					id="head"
					orient="auto"
					markerWidth={3}
					markerHeight={4}
					refX="0.1"
					refY="2"
				>
					<path d="M0,0 V4 L2,2 Z" fill="currentColor" />
				</marker>
			</defs>
			<WeightIn
				groupProps={{
					transform: "translate(0, 0)",
				}}
				titleProps={{
					x: "0.75em",
				}}
				title="conv"
				value={conv}
			/>
		</svg>
	);
}
