import { Header } from "./Header.jsx";
import { Headline } from "./Headline.jsx";
import { Main } from "./Main.jsx";

export function Metadata({ metadata, filename, worker }) {
	if (!metadata) {
		return <Main metadata={metadata} filename={filename} worker={worker} />;
	}

	return [
		<Headline key="headline" metadata={metadata} filename={filename} />,
		<Header key="header" metadata={metadata} filename={filename} />,
		<Main key="main" metadata={metadata} filename={filename} worker={worker} />,
	];
}
