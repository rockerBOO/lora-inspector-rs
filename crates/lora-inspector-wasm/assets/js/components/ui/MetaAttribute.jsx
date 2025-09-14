export function MetaAttribute({
	name,
	value,
	valueClassName,
	secondary,
	secondaryName,
	secondaryClassName,
	containerProps,
}) {
	return (
		<div {...(containerProps ?? {})}>
			<div title={name} className="caption">
				{name}
			</div>
			<div className="meta-attribute-value">
				<div title={name} className={valueClassName ?? ""}>
					{value}
				</div>

				{secondary && (
					<div className="secondary">
						<div title={secondaryName} className="caption secondary-name">
							{secondaryName}
						</div>
						<div title={name} className={secondaryClassName ?? ""}>
							{secondary}
						</div>
					</div>
				)}
			</div>
		</div>
	);
}
