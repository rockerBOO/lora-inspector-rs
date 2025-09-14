import { MetaAttribute } from "../ui/MetaAttribute.jsx";

export function ModelSpec({ metadata }) {
	const training = [
		<div className="row space-apart" key="training_timings">
			{metadata.has("ss_training_started_at") && (
				<MetaAttribute
					key="started_at"
					name="Started"
					value={new Date(
						metadata.get("ss_training_started_at") * 1000,
					).toString()}
				/>
			)}
			{metadata.has("ss_training_finished_at") && (
				<MetaAttribute
					key="finished_at"
					name="Finished"
					value={new Date(
						metadata.get("ss_training_finished_at") * 1000,
					).toString()}
				/>
			)}
			{metadata.has("ss_training_finished_at") && (
				<MetaAttribute
					key="elapsed_at"
					name="Elapsed"
					value={`${(
						(metadata.get("ss_training_finished_at") -
							metadata.get("ss_training_started_at")) /
							60
					).toPrecision(4)} minutes`}
				/>
			)}
		</div>,

		<div className="row space-apart" key="training_comments">
			{metadata.has("ss_training_comment") && (
				<div key="training_comment" className="row space-apart">
					<MetaAttribute
						name="Training comment"
						value={metadata.get("ss_training_comment")}
					/>
				</div>
			)}
		</div>,
	];

	// if (!metadata.has("modelspec.title")) {
	// 	return training;
	// }

	const img = metadata.get("modelspec.thumbnail");

	return (
		<>
			{training}
			<div className="model-spec">
				<div className="row space-apart">
					<MetaAttribute
						name="Date"
						value={new Date(metadata.get("modelspec.date")).toLocaleString()}
						key="date"
					/>
					<MetaAttribute
						name="Title"
						value={metadata.get("modelspec.title")}
						key="title"
					/>
					<MetaAttribute
						name="Author"
						value={metadata.get("modelspec.author")}
						key="description"
					/>
					<MetaAttribute
						name="Prediction type"
						value={metadata.get("modelspec.prediction_type")}
						key="prediction-type"
					/>
				</div>
				<div className="row space-apart">
					<MetaAttribute
						name="License"
						value={metadata.get("modelspec.license")}
						key="license"
					/>
					<MetaAttribute
						name="Description"
						value={metadata.get("modelspec.description")}
						key="description"
					/>
					<MetaAttribute
						name="Architecture"
						value={metadata.get("modelspec.architecture")}
						key="architecture"
					/>
					<MetaAttribute
						name="Implementation"
						value={metadata.get("modelspec.implementation")}
						key="implementation"
					/>

					<MetaAttribute
						name="Trigger Phrase"
						value={metadata.get("modelspec.trigger_phrase")}
						key="trigger_phrase"
					/>
					{img && <img src={img} alt="" />}
					<MetaAttribute
						name="Tags"
						value={metadata.get("modelspec.tags")}
						key="tags"
					/>
				</div>

				{metadata.has("modelspec.hash_sha256") && (
					<MetaAttribute
						name="SHA256"
						value={metadata.get("modelspec.hash_sha256")}
						key={sha256}
					/>
				)}
				{metadata.has("ss_training_comment") && (
					<div className="row space-apart">
						<MetaAttribute
							name="Training comment"
							value={metadata.get("ss_training_comment")}
						/>
					</div>
				)}
			</div>
		</>
	);
}
