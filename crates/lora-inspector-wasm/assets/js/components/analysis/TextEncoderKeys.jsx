import React from "react";

export function TextEncoderKeys({ textEncoderKeys }) {
	return [
		<h3 key="text-encoder-keys-header">Text encoder keys</h3>,
		<ul key="text-encoder-keys">
			{textEncoderKeys.map((textEncoderKey) => {
				return <li key={textEncoderKey}>{textEncoderKey}</li>;
			})}
		</ul>,
	];
}
