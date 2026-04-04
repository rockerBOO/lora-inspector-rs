import { Header } from "./Header.jsx";
import { Headline } from "./Headline.jsx";
import { Main } from "./Main.jsx";

export function Metadata({ metadata, filename, worker, nav }) {
	if (!metadata) {
		return (
			<>
				{nav}
				<Main metadata={metadata} filename={filename} worker={worker} />
			</>
		);
	}

	return (
		<>
			{nav}
			<Headline metadata={metadata} filename={filename} />
			<Header metadata={metadata} filename={filename} />
			<Main metadata={metadata} filename={filename} worker={worker} />
		</>
	);
}
