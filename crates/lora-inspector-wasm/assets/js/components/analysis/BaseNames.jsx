export function BaseNames({ baseNames }) {
	return [
		<h3 key="header-base-names">Base names</h3>,
		<ul key="base-names">
			{baseNames.map((baseName) => {
				return <li key={baseName}>{baseName}</li>;
			})}
		</ul>,
	];
}
