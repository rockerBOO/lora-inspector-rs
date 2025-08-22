import React from "react";

export function UnetKeys({ unetKeys }) {
	return [
		<h3 key="unet-keys-header">UNet keys</h3>,
		<ul key="unet-keys">
			{unetKeys.map((unetKey) => {
				return <li key={unetKey}>{unetKey}</li>;
			})}
		</ul>,
	];
}
