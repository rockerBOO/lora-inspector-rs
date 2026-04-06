import { Headline } from "./Headline.jsx";
import { Main } from "./Main.jsx";

export function Metadata({ metadata, filename, worker, nav }) {
	return (
		<>
			{nav}
			{filename && <Headline metadata={metadata} filename={filename} />}
			<Main metadata={metadata} filename={filename} worker={worker} />
		</>
	);
}
