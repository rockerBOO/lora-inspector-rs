import React from "react";
import { MetaAttribute } from "../ui/MetaAttribute.jsx";

export function VAE({ metadata }) {
	if (!metadata.has("ss_vae_name")) {
		return null;
	}

	return (
		<div>
			<div>
				<MetaAttribute name="VAE name" value={metadata.get("ss_vae_name")} />
				<MetaAttribute
					name="VAE hash"
					valueClassName="hash"
					value={metadata.get("ss_vae_hash")}
				/>
				<MetaAttribute
					name="New VAE hash"
					valueClassName="hash"
					value={metadata.get("ss_new_vae_hash")}
				/>
			</div>
		</div>
	);
}
