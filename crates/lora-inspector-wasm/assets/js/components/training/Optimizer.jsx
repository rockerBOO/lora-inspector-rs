import { MetaAttribute } from "../ui/MetaAttribute.jsx";

function parseOptimizer(value) {
	if (!value) return null;
	const parenOpen = value.indexOf("(");
	const parenClose = value.lastIndexOf(")");
	if (parenOpen === -1 || parenClose === -1) return null;

	const fullPath = value.slice(0, parenOpen);
	const kwargsStr = value.slice(parenOpen + 1, parenClose);
	const className = fullPath.split(".").pop();

	const kwargs = Object.fromEntries(
		kwargsStr
			.split(",")
			.map((kv) => kv.split("="))
			.filter(([k]) => k?.trim())
			.map(([k, v]) => [k.trim(), v?.trim()]),
	);

	return { fullPath, className, kwargs };
}

export function Optimizer({ metadata }) {
	const optimizerRaw = metadata.get("ss_optimizer");
	const parsed = parseOptimizer(optimizerRaw);

	return (
		<>
			<div className="row space-apart">
				<MetaAttribute
					name="Optimizer"
					containerProps={{ className: "span-3" }}
					value={parsed ? parsed.className : optimizerRaw}
				/>
				<MetaAttribute
					name="Seed"
					valueClassName="number"
					value={metadata.get("ss_seed")}
				/>
			</div>
			{parsed && (
				<>
					<MetaAttribute
						name="Full optimizer path"
						value={parsed.fullPath}
						valueClassName="args"
					/>
					{Object.keys(parsed.kwargs).length > 0 && (
						<div className="row space-apart">
							{Object.entries(parsed.kwargs).map(([k, v]) => (
								<MetaAttribute key={k} name={k} value={v} />
							))}
						</div>
					)}
				</>
			)}
		</>
	);
}
