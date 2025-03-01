import { createWriteStream } from "node:fs";
import { Readable } from "node:stream";

async function globalSetup(_config) {
	const url =
		"https://lora-inspector-test-files.us-east-1.linodeobjects.com/boo.safetensors";
	const fileName = url.split("/").pop();
	const resp = await fetch(url);
	if (resp.ok && resp.body) {
		console.log("Writing to file:", fileName);
		const writer = createWriteStream(fileName);
		Readable.fromWeb(resp.body).pipe(writer);
	} else {
		console.error("could not download the file");
	}
}

export default globalSetup;
