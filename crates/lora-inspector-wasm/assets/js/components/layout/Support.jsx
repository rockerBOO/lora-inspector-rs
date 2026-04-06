/** biome-ignore-all lint/a11y/noStaticElementInteractions: Having this overlay be clickable is temporary */
import { useState } from "react";

export function Support() {
	const [modal, setModal] = useState(false);

	if (modal) {
		return (
			<div
				className="modal-overlay"
				onClick={() => setModal(false)}
				onKeyDown={(e) => e.key === "Escape" && setModal(false)}
			>
				<div
					className="modal"
					onClick={(e) => e.stopPropagation()}
					onKeyDown={(e) => e.stopPropagation()}
				>
					<div className="modal-header">
						<h2>LoRA Inspector Support</h2>
						<button
							type="button"
							className="modal-close"
							onClick={() => setModal(false)}
							aria-label="Close"
						>
							×
						</button>
					</div>
					<div className="modal-content">
						<p className="modal-description">
							LoRA Inspector is a free, open-source tool for analyzing LoRA
							files.
						</p>
						<section className="modal-section">
							<h3>Contribute</h3>
							<ul>
								<li>
									<a
										href="https://github.com/rockerBOO/lora-inspector-rs/issues"
										target="_blank"
										rel="noopener noreferrer"
									>
										Report bugs or suggest features
									</a>
								</li>
							</ul>
						</section>
						<section className="modal-section">
							<h3>Support the project</h3>
							<ul>
								<li>
									<a
										href="https://github.com/rockerBOO/lora-inspector-rs"
										target="_blank"
										rel="noopener noreferrer"
									>
										Star on GitHub
									</a>
								</li>
								<li>
									<a
										href="https://ko-fi.com/rockerboo"
										target="_blank"
										rel="noopener noreferrer"
									>
										Buy me a coffee
									</a>
								</li>
							</ul>
						</section>
					</div>
				</div>
			</div>
		);
	}

	return (
		<button
			type="button"
			className="support-button"
			onClick={() => {
				setModal(true);
			}}
		>
			Support
		</button>
	);
}
