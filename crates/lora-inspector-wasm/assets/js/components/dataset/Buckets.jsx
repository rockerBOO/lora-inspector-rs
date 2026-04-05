import { MetaAttribute } from "../ui/MetaAttribute.jsx";
import { BucketInfo } from "./BucketInfo.jsx";
import { Subset } from "./Subset.jsx";
import { TagFrequency } from "./TagFrequency.jsx";

function formatResolution(resolution) {
	if (Array.isArray(resolution)) return `${resolution[0]}x${resolution[1]}`;
	return resolution;
}

function formatArray(arr) {
	if (Array.isArray(arr)) return arr.join(", ");
	return arr;
}

const VIDEO_KEYS = [
	"target_frames",
	"frame_extraction",
	"frame_stride",
	"frame_sample",
	"max_frames",
	"source_fps",
];

const FRAMEPACK_KEYS = [
	"fp_latent_window_size",
	"fp_1f_clean_indices",
	"fp_1f_target_index",
	"fp_1f_no_post",
];

const SOURCE_KEYS = [
	"image_directory",
	"video_directory",
	"image_jsonl_file",
	"video_jsonl_file",
];

export function Buckets({ dataset, metadata }) {
	const hasSource = SOURCE_KEYS.some((k) => k in dataset);
	const hasVideoFields = VIDEO_KEYS.some((k) => k in dataset);
	const hasFramePackFields = FRAMEPACK_KEYS.some((k) => k in dataset);
	const hasControlFields =
		"control_directory" in dataset ||
		"control_resolution" in dataset ||
		"no_resize_control" in dataset;

	return [
		// Source row
		hasSource && (
			<div key="source" className="row-compact">
				{"image_directory" in dataset && (
					<MetaAttribute
						name="Image directory"
						value={dataset.image_directory}
					/>
				)}
				{"video_directory" in dataset && (
					<MetaAttribute
						name="Video directory"
						value={dataset.video_directory}
					/>
				)}
				{"image_jsonl_file" in dataset && (
					<MetaAttribute name="Image JSONL" value={dataset.image_jsonl_file} />
				)}
				{"video_jsonl_file" in dataset && (
					<MetaAttribute name="Video JSONL" value={dataset.video_jsonl_file} />
				)}
			</div>
		),

		// Bucket + repeats row
		<div key="buckets" className="row-compact">
			<MetaAttribute
				name="Buckets"
				value={dataset.enable_bucket ? "True" : "False"}
			/>
			{"bucket_no_upscale" in dataset && (
				<MetaAttribute
					name="No upscale"
					value={dataset.bucket_no_upscale ? "True" : "False"}
				/>
			)}
			{"min_bucket_reso" in dataset && (
				<MetaAttribute
					name="Min bucket resolution"
					valueClassName="number"
					value={dataset.min_bucket_reso}
				/>
			)}
			{"max_bucket_reso" in dataset && (
				<MetaAttribute
					name="Max bucket resolution"
					valueClassName="number"
					value={dataset.max_bucket_reso}
				/>
			)}
			{"resolution" in dataset && (
				<MetaAttribute
					name="Resolution"
					valueClassName="number"
					value={formatResolution(dataset.resolution)}
				/>
			)}
			{"num_repeats" in dataset && (
				<MetaAttribute
					name="Repeats"
					valueClassName="number"
					value={dataset.num_repeats}
				/>
			)}
			{"caption_extension" in dataset && (
				<MetaAttribute
					name="Caption extension"
					value={dataset.caption_extension}
				/>
			)}
			{"cache_directory" in dataset && (
				<MetaAttribute
					name="Cache directory"
					value={dataset.cache_directory}
				/>
			)}
		</div>,

		// Video row
		hasVideoFields && (
			<div key="video" className="row-compact">
				{"target_frames" in dataset && (
					<MetaAttribute
						name="Target frames"
						valueClassName="number"
						value={formatArray(dataset.target_frames)}
					/>
				)}
				{"frame_extraction" in dataset && (
					<MetaAttribute
						name="Frame extraction"
						value={dataset.frame_extraction}
					/>
				)}
				{"frame_stride" in dataset && (
					<MetaAttribute
						name="Frame stride"
						valueClassName="number"
						value={dataset.frame_stride}
					/>
				)}
				{"frame_sample" in dataset && (
					<MetaAttribute
						name="Frame sample"
						valueClassName="number"
						value={dataset.frame_sample}
					/>
				)}
				{"max_frames" in dataset && (
					<MetaAttribute
						name="Max frames"
						valueClassName="number"
						value={dataset.max_frames}
					/>
				)}
				{"source_fps" in dataset && (
					<MetaAttribute
						name="Source FPS"
						valueClassName="number"
						value={dataset.source_fps}
					/>
				)}
			</div>
		),

		// Control row — only shown when there is actual control data
		(hasControlFields || dataset.has_control === true) && (
			<div key="control" className="row-compact">
				{"control_directory" in dataset && (
					<MetaAttribute
						name="Control directory"
						value={dataset.control_directory}
					/>
				)}
				{"control_resolution" in dataset && (
					<MetaAttribute
						name="Control resolution"
						valueClassName="number"
						value={formatArray(dataset.control_resolution)}
					/>
				)}
				{"no_resize_control" in dataset && (
					<MetaAttribute
						name="No resize control"
						value={dataset.no_resize_control ? "True" : "False"}
					/>
				)}
				{!hasControlFields && dataset.has_control === true && (
					<MetaAttribute name="Has control" value="True" />
				)}
			</div>
		),

		// FramePack row
		hasFramePackFields && (
			<div key="framepack" className="row-compact">
				{"fp_latent_window_size" in dataset && (
					<MetaAttribute
						name="Latent window size"
						valueClassName="number"
						value={dataset.fp_latent_window_size}
					/>
				)}
				{"fp_1f_clean_indices" in dataset && (
					<MetaAttribute
						name="1F clean indices"
						value={formatArray(dataset.fp_1f_clean_indices)}
					/>
				)}
				{"fp_1f_target_index" in dataset && (
					<MetaAttribute
						name="1F target index"
						valueClassName="number"
						value={dataset.fp_1f_target_index}
					/>
				)}
				{"fp_1f_no_post" in dataset && (
					<MetaAttribute
						name="1F no post"
						value={dataset.fp_1f_no_post ? "True" : "False"}
					/>
				)}
			</div>
		),

		// Special row
		"multiple_target" in dataset && (
			<div key="special" className="row-compact">
				<MetaAttribute
					name="Multiple target"
					value={dataset.multiple_target ? "True" : "False"}
				/>
			</div>
		),

		// Bucket info
		dataset.bucket_info && (
			<div key="bucket-info">
				<BucketInfo metadata={metadata} dataset={dataset} />
			</div>
		),

		// Subsets
		...("subsets" in dataset
			? [
					<h3 key="subsets-header">Subsets:</h3>,
					<div key="subsets" className="subsets">
						{dataset.subsets.map((subset, i) => (
							<Subset
								key={`subset-${subset.image_dir}-${i}`}
								metadata={metadata}
								subset={subset}
							/>
						))}
					</div>,
				]
			: []),

		// Tag frequencies
		...(Object.keys(dataset?.tag_frequency ?? {}).length > 0
			? [
					<h3 key="header-tag-frequencies">Tag frequencies</h3>,
					<div
						key="tag-frequencies"
						className="tag-frequencies row space-apart"
					>
						{Object.entries(dataset.tag_frequency).map(([dir, frequency]) => (
							<div key={dir}>
								<h3>{dir}</h3>
								<TagFrequency
									key="tag-frequency"
									tagFrequency={frequency}
									metadata={metadata}
								/>
							</div>
						))}
					</div>,
				]
			: []),
	].filter(Boolean);
}
