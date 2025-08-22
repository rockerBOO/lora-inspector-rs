import React, { useState, useEffect, createRef, useCallback } from "react";
import { trySyncMessage } from "../../message.js";
import { BaseNames } from "./BaseNames.jsx";
import { UnetKeys } from "./UnetKeys.jsx";
import { TextEncoderKeys } from "./TextEncoderKeys.jsx";
import { AllKeys } from "./AllKeys.jsx";
import { Statistics } from "./Statistics.jsx";

const DEBUG = new URLSearchParams(document.location.search).has("DEBUG");
const VERBOSE = new URLSearchParams(document.location.search).has("VERBOSE");

export function Advanced({ filename, worker }) {
	const [baseNames, setBaseNames] = useState([]);
	const [showBaseNames, setShowBlockNames] = useState(false);

	const [textEncoderKeys, setTextEncoderKeys] = useState([]);
	const [showTextEncoderKeys, setShowTextEncoderKeys] = useState(false);

	const [unetKeys, setUnetKeys] = useState([]);
	const [showUnetKeys, setShowUnetKeys] = useState(false);

	const [allKeys, setAllKeys] = useState([]);
	const [showAllKeys, setShowAllKeys] = useState(false);

	const [canHaveStatistics, setCanHaveStatistics] = useState(false);
	const [networkType, setNetworkType] = useState(null);

	// Debugging state
	const [debugMessages, setDebugMessages] = useState([]);

	const addDebugMessage = (message) => {
		if (DEBUG) {
			setDebugMessages((prev) => [...prev, message]);
		}
	};

	const advancedRef = createRef();

	// Worker validation
	const validateWorker = useCallback(() => {
		if (!worker) {
			const errorMsg = `Worker is undefined for file: ${filename}`;
			addDebugMessage(errorMsg);
			console.error(errorMsg);
			return false;
		}
		return true;
	}, [filename, worker]);

	useEffect(() => {
		trySyncMessage({ messageType: "base_names", name: filename }, worker)
			.then((resp) => {
				if (resp.baseNames && resp.baseNames.length > 0) {
					resp.baseNames.sort();
					setBaseNames(resp.baseNames);
					if (VERBOSE)
						addDebugMessage(`Found ${resp.baseNames.length} base names`);
				} else {
					addDebugMessage(`No base names found for file: ${filename}`);
					console.warn(`No base names found for file: ${filename}`);
				}
			})
			.catch((error) => {
				const errorMsg = `Error extracting base names: ${error}`;
				addDebugMessage(errorMsg);
				console.error(errorMsg);
			});

		trySyncMessage({ messageType: "text_encoder_keys", name: filename }, worker)
			.then((resp) => {
				if (resp.textEncoderKeys && resp.textEncoderKeys.length > 0) {
					resp.textEncoderKeys.sort();
					setTextEncoderKeys(resp.textEncoderKeys);
					if (VERBOSE)
						addDebugMessage(
							`Found ${resp.textEncoderKeys.length} text encoder keys`,
						);
				} else {
					addDebugMessage(`No text encoder keys found for file: ${filename}`);
					console.warn(`No text encoder keys found for file: ${filename}`);
				}
			})
			.catch((error) => {
				const errorMsg = `Error extracting text encoder keys: ${error}`;
				addDebugMessage(errorMsg);
				console.error(errorMsg);
			});

		trySyncMessage({ messageType: "unet_keys", name: filename }, worker)
			.then((resp) => {
				if (resp.unetKeys && resp.unetKeys.length > 0) {
					resp.unetKeys.sort();
					setUnetKeys(resp.unetKeys);
					if (VERBOSE)
						addDebugMessage(`Found ${resp.unetKeys.length} UNet keys`);
				} else {
					addDebugMessage(`No UNet keys found for file: ${filename}`);
					console.warn(`No UNet keys found for file: ${filename}`);
				}
			})
			.catch((error) => {
				const errorMsg = `Error extracting UNet keys: ${error}`;
				addDebugMessage(errorMsg);
				console.error(errorMsg);
			});

		trySyncMessage({ messageType: "keys", name: filename }, worker)
			.then((resp) => {
				if (resp.keys && resp.keys.length > 0) {
					resp.keys.sort();
					setAllKeys(resp.keys);
					if (VERBOSE) addDebugMessage(`Found ${resp.keys.length} total keys`);
				} else {
					addDebugMessage(`No keys found for file: ${filename}`);
					console.warn(`No keys found for file: ${filename}`);
				}
			})
			.catch((error) => {
				const errorMsg = `Error extracting keys: ${error}`;
				addDebugMessage(errorMsg);
				console.error(errorMsg);
			});
	}, [filename, worker]);

	useEffect(() => {
		trySyncMessage(
			{
				messageType: "network_type",
				name: filename,
				reply: true,
			},
			worker,
		)
			.then((resp) => {
				const supportedTypes = [
					"LoRA",
					"LoRAFA",
					"DyLoRA",
					"LoHA",
					"LoKr",
					"OFT",
					"BOFT",
					"DoRA",
					"KohyaSSLoRA",
					undefined,
				];

				setNetworkType(resp.networkType);

				if (supportedTypes.includes(resp.networkType)) {
					setCanHaveStatistics(true);
					addDebugMessage(`Network type detected: ${resp.networkType}`);

					trySyncMessage(
						{
							messageType: "precision",
							name: filename,
							reply: true,
						},
						worker,
					).then((precResp) => {
						if (precResp.precision === "bf16") {
							setCanHaveStatistics(false);
							addDebugMessage(`Precision bf16 detected: disabling statistics`);
						}
					});
				} else {
					addDebugMessage(`Unsupported network type: ${resp.networkType}`);
				}
			})
			.catch((error) => {
				const errorMsg = `Error determining network type: ${error}`;
				addDebugMessage(errorMsg);
				console.error(errorMsg);
			});
	}, [filename, worker]);

	useEffect(() => {
		if (DEBUG && advancedRef.current) {
			advancedRef.current.scrollIntoView({ behavior: "smooth" });
		}
	}, [advancedRef.current]);

	return (
		<>
			<h2 id="advanced" ref={advancedRef}>
				Advanced
			</h2>
			{DEBUG && debugMessages.length > 0 && (
				<div
					className="debug-info"
					style={{
						backgroundColor: "#f0f0f0",
						padding: "10px",
						marginBottom: "10px",
						borderRadius: "5px",
					}}
				>
					<h3>Debug Information</h3>
					{debugMessages.map((msg, index) => (
						<p key={index} style={{ margin: "5px 0", color: "darkred" }}>
							{msg}
						</p>
					))}
				</div>
			)}
			<div className="row">
				<div>
					{showBaseNames ? (
						<BaseNames baseNames={baseNames} />
					) : (
						<div>Base name keys: {baseNames.length}</div>
					)}
				</div>
				<div>
					{showTextEncoderKeys ? (
						<TextEncoderKeys textEncoderKeys={textEncoderKeys} />
					) : (
						<div>Text encoder keys: {textEncoderKeys.length}</div>
					)}
				</div>
				<div>
					{showUnetKeys ? (
						<UnetKeys unetKeys={unetKeys} />
					) : (
						<div>Unet keys: {unetKeys.length}</div>
					)}
				</div>
				<div>
					{showAllKeys ? (
						<AllKeys allKeys={allKeys} />
					) : (
						<div>All keys: {allKeys.length}</div>
					)}
				</div>
			</div>
			{!canHaveStatistics ? (
				<BaseNames baseNames={baseNames} />
			) : (
				<Statistics baseNames={baseNames} filename={filename} worker={worker} />
			)}
		</>
	);
}
