export function Group(props) {
	return <g {...props} />;
}

export function Line({ d, ...rest }) {
	return (
		<path
			markerEnd="none"
			strokeWidth={4}
			fill="none"
			stroke="currentColor"
			d={d}
			{...rest}
		/>
	);
}

export function LineEnd(props) {
	return <Line markerEnd="url(#head)" {...props} />;
}

export function GText({ children, groupProps, ...rest }) {
	return (
		<g {...groupProps}>
			<text {...rest}>{children}</text>
		</g>
	);
}

export function WeightIn({ groupProps, titleProps, valueProps, title, value }) {
	return (
		<g {...groupProps}>
			<path
				markerEnd="url(#head)"
				strokeWidth={4}
				fill="none"
				stroke="currentColor"
				d="M36,0 36,10"
			/>
			<text {...(titleProps ?? {})} title={title} y="2em">
				{title}
			</text>
			<text x="0.5em" y="3.5em" {...(valueProps ?? {})} title={title}>
				{value?.toPrecision(4)}
			</text>
		</g>
	);
}

export function SimpleWeight({
	groupProps,
	titleProps,
	valueProps,
	title,
	value,
}) {
	return (
		<g className={title} {...groupProps}>
			<text x="0.5em" {...(titleProps ?? {})} title={title}>
				{title}
			</text>
			<text x="0.5em" y="1.5em" {...(valueProps ?? {})} title={title}>
				{value?.toPrecision(4)}
			</text>
		</g>
	);
}
