import { MetaAttribute } from "../ui/MetaAttribute.jsx";

export function Subset({ subset }) {
	return (
		<div className="subset">
			<h4>{subset.image_dir}</h4>
			<div className="row-compact">
				{"class_tokens" in subset && (
					<MetaAttribute name="Class tokens" value={subset.class_tokens} />
				)}
				{"num_repeats" in subset && (
					<MetaAttribute
						name="Repeats"
						valueClassName="number"
						value={subset.num_repeats}
					/>
				)}
				{"keep_tokens" in subset && (
					<MetaAttribute
						name="Keep tokens"
						valueClassName="number"
						value={subset.keep_tokens}
					/>
				)}
				{"shuffle_caption" in subset && (
					<MetaAttribute
						name="Shuffle caption"
						value={subset.shuffle_caption ? "True" : "False"}
					/>
				)}
				{"flip_aug" in subset && (
					<MetaAttribute
						name="Flip aug"
						value={subset.flip_aug ? "True" : "False"}
					/>
				)}
				{"color_aug" in subset && (
					<MetaAttribute
						name="Color aug"
						value={subset.color_aug ? "True" : "False"}
					/>
				)}
				{"random_crop" in subset && (
					<MetaAttribute
						name="Random crop"
						value={subset.random_crop ? "True" : "False"}
					/>
				)}
				{"is_reg" in subset && (
					<MetaAttribute
						name="Regularization"
						value={subset.is_reg ? "True" : "False"}
					/>
				)}
				{"caption_extension" in subset && (
					<MetaAttribute
						name="Caption extension"
						value={subset.caption_extension}
					/>
				)}
				{"caption_prefix" in subset && (
					<MetaAttribute name="Caption prefix" value={subset.caption_prefix} />
				)}
				{"caption_suffix" in subset && (
					<MetaAttribute name="Caption suffix" value={subset.caption_suffix} />
				)}
				{"keep_tokens_separator" in subset && (
					<MetaAttribute
						name="Keep tokens separator"
						value={subset.keep_tokens_separator}
					/>
				)}
				{"secondary_separator" in subset && (
					<MetaAttribute
						name="Secondary separator"
						value={subset.secondary_separator}
					/>
				)}
				{"enable_wildcard" in subset && (
					<MetaAttribute
						name="Wildcard"
						value={subset.enable_wildcard ? "True" : "False"}
					/>
				)}
				{"face_crop_aug_range" in subset && (
					<MetaAttribute
						name="Face crop aug range"
						value={subset.face_crop_aug_range.join(", ")}
					/>
				)}
			</div>
		</div>
	);
}
