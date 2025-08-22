import React, { useState } from "react";

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
						<h2>Support LoRA Inspector</h2>
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
						<p>
							LoRA Inspector is a free, open-source tool for analyzing LoRA
							files. If you find it useful, consider supporting its development:
						</p>
						<ul>
							<li>
								<a
									href="https://github.com/rockerBOO/lora-inspector-rs"
									target="_blank"
									rel="noopener noreferrer"
								>
									⭐ Star the project on GitHub
								</a>
							</li>
							<li>
								<a
									href="https://github.com/rockerBOO/lora-inspector-rs/issues"
									target="_blank"
									rel="noopener noreferrer"
								>
									🐛 Report bugs or suggest features
								</a>
							</li>
							<li>
								<a
									href="https://ko-fi.com/rockerboo"
									target="_blank"
									rel="noopener noreferrer"
								>
									☕ Buy me a coffee
								</a>
							</li>
						</ul>
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
