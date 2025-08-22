import React from "react";

export function AllKeys({ allKeys }) {
	return [
		<h3 key="all-keys-header">All keys</h3>,
		<ul key="all-keys">
			{allKeys.map((key) => {
				return <li key={key}>{key}</li>;
			})}
		</ul>,
	];
}
