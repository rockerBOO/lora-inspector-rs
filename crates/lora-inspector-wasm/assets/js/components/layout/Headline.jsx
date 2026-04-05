import { Raw } from "./Raw.jsx";
import { SectionNav } from "./SectionNav.jsx";

export function Headline({ metadata, filename }) {
	return (
		<>
			<div className="file-header">
				<div>
					<div className="file-label">LoRA file</div>
					<h1>{filename}</h1>
				</div>
				{metadata && <Raw metadata={metadata} filename={filename} />}
			</div>
			<SectionNav filename={filename} />
		</>
	);
}
