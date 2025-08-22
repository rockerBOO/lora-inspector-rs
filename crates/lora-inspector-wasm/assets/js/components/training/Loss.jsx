import React from "react";
import { MetaAttribute } from "../ui/MetaAttribute.jsx";

export function Loss({ metadata }) {
	return (
		<div className="row space-apart">
			<MetaAttribute
				name="Loss type"
				valueClassName="attribute"
				value={metadata.get("ss_loss_type")}
			/>
			{metadata.get("ss_loss_type") === "huber" && (
				<MetaAttribute
					name="Huber schedule"
					valueClassName="attribute"
					value={metadata.get("ss_huber_schedule")}
				/>
			)}
			{metadata.get("ss_loss_type") === "huber" && (
				<MetaAttribute
					name="Huber c"
					valueClassName="number"
					value={metadata.get("ss_huber_c")}
				/>
			)}
			{metadata.get("ss_loss_type") === "huber" && (
				<MetaAttribute
					name="Huber scale"
					valueClassName="number"
					value={metadata.get("ss_huber_scale") ?? 1.0}
				/>
			)}
			<MetaAttribute
				name="Debiased Estimation"
				valueClassName="boolean"
				value={metadata.get("ss_debiased_estimation") ?? "False"}
			/>
			<MetaAttribute
				name="Min SNR Gamma"
				valueClassName="number"
				value={metadata.get("ss_min_snr_gamma")}
			/>
			<MetaAttribute
				name="Zero Terminal SNR"
				valueClassName="boolean"
				value={metadata.get("ss_zero_terminal_snr")}
			/>
			{metadata.has("ss_masked_loss") !== undefined && (
				<MetaAttribute
					name="Masked Loss"
					value={metadata.get("ss_masked_loss")}
				/>
			)}
		</div>
	);
}
