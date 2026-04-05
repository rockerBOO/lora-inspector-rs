export function Section({ id, label, children }) {
	return (
		<section id={id}>
			<h2>{label}</h2>
			{children}
		</section>
	);
}
