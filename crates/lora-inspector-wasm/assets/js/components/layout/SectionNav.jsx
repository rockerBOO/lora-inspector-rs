import { useEffect, useRef, useState } from "react";

const SECTIONS = [
	{ id: "metadata", label: "Metadata" },
	{ id: "network", label: "Network" },
	{ id: "training", label: "Training" },
	{ id: "optimizer", label: "Optimizer" },
	{ id: "dataset", label: "Dataset" },
	{ id: "advanced", label: "Advanced" },
];

export function SectionNav({ filename }) {
	const [activeId, setActiveId] = useState("metadata");
	const navRef = useRef(null);

	useEffect(() => {
		const observer = new IntersectionObserver(
			(entries) => {
				for (const entry of entries) {
					if (entry.isIntersecting) {
						setActiveId(entry.target.id);
					}
				}
			},
			{
				rootMargin: "-10% 0px -80% 0px",
				threshold: 0,
			},
		);

		for (const { id } of SECTIONS) {
			const el = document.getElementById(id);
			if (el) observer.observe(el);
		}

		return () => observer.disconnect();
	}, []);

	return (
		<div className="section-nav-wrapper" ref={navRef}>
			<span className="file-bar-name">{filename}</span>
			<nav aria-label="Page sections">
				<ul>
					{SECTIONS.map(({ id, label }) => (
						<li key={id}>
							<a
								href={`#${id}`}
								aria-current={activeId === id ? "true" : undefined}
							>
								{label}
							</a>
						</li>
					))}
				</ul>
			</nav>
			<button
				type="button"
				aria-label="Back to top"
				className="to-top"
				onClick={() => window.scrollTo({ top: 0, behavior: "smooth" })}
			>
				↑
			</button>
		</div>
	);
}
