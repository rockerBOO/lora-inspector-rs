import { LineChart, easings } from "chartist";
import React from "react";
import ReactDOM from "react-dom";
import init from "/pkg";

const h = React.createElement;

function Header({ metadata }) {
  return h("header", null, h(ModelSpec, { metadata }));
}

function ModelSpec({ metadata }) {
  let training = [
    h("div", { className: "row space-apart", key: "training_timings" }, [
      metadata.has("ss_training_started_at") &&
        h(MetaAttribute, {
          key: "started_at",
          name: "Started",
          value: new Date(
            metadata.get("ss_training_started_at") * 1000,
          ).toString(),
        }),
      metadata.has("ss_training_finished_at") &&
        h(MetaAttribute, {
          key: "finished_at",
          name: "Finished",
          value: new Date(
            metadata.get("ss_training_finished_at") * 1000,
          ).toString(),
        }),
      metadata.has("ss_training_finished_at") &&
        h(MetaAttribute, {
          key: "elapsed_at",
          name: "Elapsed",
          value:
            (
              (metadata.get("ss_training_finished_at") -
                metadata.get("ss_training_started_at")) /
              60
            ).toPrecision(4) + " minutes",
        }),
    ]),

    h(
      "div",
      { className: "row space-apart", key: "training_comments" },
      metadata.has("ss_training_comment") &&
        h(
          "div",
          {
            key: "training_comment",
            className: "row space-apart",
          },
          h(MetaAttribute, {
            name: "Training comment",
            value: metadata.get("ss_training_comment"),
          }),
        ),
    ),
  ];

  if (!metadata.has("modelspec.title")) {
    return training;
  }

  return h(
    "div",
    { className: "model-spec" },
    h("div", { className: "row space-apart" }, [
      h(MetaAttribute, {
        name: "Date",
        value: new Date(metadata.get("modelspec.date")).toLocaleString(),
        key: "date",
      }),
      h(MetaAttribute, {
        name: "Title",
        value: metadata.get("modelspec.title"),
        key: "title",
      }),
      h(MetaAttribute, {
        name: "Prediction type",
        value: metadata.get("modelspec.prediction_type"),
        key: "prediction-type",
      }),
    ]),
    h("div", { className: "row space-apart" }, [
      h(MetaAttribute, {
        name: "License",
        value: metadata.get("modelspec.license"),
        key: "license",
      }),
      h(MetaAttribute, {
        name: "Description",
        value: metadata.get("modelspec.description"),
        key: "description",
      }),

      h(MetaAttribute, {
        name: "Tags",
        value: metadata.get("modelspec.tags"),

        key: "tags",
      }),
    ]),
    training,
  );
}

function PretrainedModel({ metadata }) {
  return h(
    "div",
    { className: "pretrained-model row space-apart" }, //
    // h(MetaAttribute, { className: "caption" }, "SD Model"),
    h(MetaAttribute, {
      name: "SD model name",
      value: metadata.get("ss_sd_model_name"),
    }),
    h(
      "div",
      {},
      h(MetaAttribute, {
        name: "Model hash",
        value: metadata.get("sshs_model_hash"),
        valueClassName: "hash",
        metadata,
      }),
      h(MetaAttribute, {
        name: "Legacy model hash",
        value: metadata.get("sshs_legacy_hash"),
        metadata,
      }),
    ),
    h(
      "div",
      {},
      h(MetaAttribute, {
        name: "Session ID",
        value: metadata.get("ss_session_id"),
      }),
      h(MetaAttribute, {
        name: "sd-scripts commit hash",
        value: metadata.get("ss_sd_scripts_commit_hash"),
        valueClassName: "hash",
      }),
    ),
    h("div", null, h(VAE, { metadata })),
  );
}

function VAE({ metadata }) {
  if (!metadata.has("ss_vae_name")) {
    return null;
  }

  return h(
    "div",
    null,
    h(
      "div",
      null,
      h(MetaAttribute, {
        name: "VAE name",
        value: metadata.get("ss_vae_name"),
      }),
      h(MetaAttribute, {
        name: "VAE hash",
        valueClassName: "hash",
        value: metadata.get("ss_vae_hash"),
      }),
      h(MetaAttribute, {
        name: "New VAE hash",
        valueClassName: "hash",
        value: metadata.get("ss_new_vae_hash"),
      }),
    ),
  );
}

function MetaAttribute({
  name,
  value,
  valueClassName,
  secondary,
  secondaryName,
  secondaryClassName,
  containerProps,
}) {
  return h(
    "div",
    containerProps ?? null,
    h("div", { title: name, className: "caption" }, name),
    h(
      "div",
      { className: "meta-attribute-value" },
      h("div", { title: name, className: valueClassName ?? "" }, value),

      secondary &&
        h(
          "div",
          { className: "secondary" },
          h(
            "div",
            { title: secondaryName, className: "caption secondary-name" },
            secondaryName,
          ),
          h(
            "div",
            { title: name, className: secondaryClassName ?? "" },
            secondary,
          ),
        ),
    ),
  );
}

function supportsDoRA(networkType) {
  return (
    networkType === "LoRA" ||
    networkType === "LoHa" ||
    networkType === "LoRAFA" ||
    networkType === "LoKr" ||
    networkType === "GLoRA"
  );
}

function Network({ metadata }) {
  const [networkModule, setNetworkModule] = React.useState(
    metadata.get("ss_network_module"),
  );
  const [networkType, setNetworkType] = React.useState("");
  const [networkArgs, setNetworkArgs] = React.useState(
    metadata.has("ss_network_args")
      ? JSON.parse(metadata.get("ss_network_args"))
      : null,
  );
  const [weightDecomposition, setWeightDecomposition] = React.useState(null);
  const [rankStabilized, setRankStabilized] = React.useState(false);

  // // TODO: date parsing isn't working right or the date is invalid
  // const trainingStart = new Date(Number.parseInt(metadata.get("ss_training_started_at"))).toLocaleString()
  // const trainingEnded = new Date(Number.parseInt(metadata.get("ss_training_ended_at"))).toLocaleString()

  React.useEffect(() => {
    trySyncMessage(
      { messageType: "network_args", name: mainFilename },
      mainFilename,
    ).then((resp) => {
      setNetworkArgs(resp.networkArgs);
    });
    trySyncMessage(
      { messageType: "network_module", name: mainFilename },
      mainFilename,
    ).then((resp) => {
      setNetworkModule(resp.networkModule);
    });
    trySyncMessage(
      { messageType: "network_type", name: mainFilename },
      mainFilename,
    ).then((resp) => {
      setNetworkType(resp.networkType);
    });
    trySyncMessage(
      { messageType: "weight_decomposition", name: mainFilename },
      mainFilename,
    ).then((resp) => {
      setWeightDecomposition(resp.weightDecomposition);
    });
    trySyncMessage(
      { messageType: "rank_stabilized", name: mainFilename },
      mainFilename,
    ).then((resp) => {
      setRankStabilized(resp.rankStabilized);
    });
  }, []);

  let networkOptions;

  if (networkType === "DiagOFT") {
    networkOptions = h(DiagOFTNetwork, { metadata });
  } else if (networkType === "BOFT") {
    networkOptions = h(BOFTNetwork, { metadata });
  } else if (networkType === "LoKr") {
    networkOptions = h(LoKrNetwork, { metadata });
  } else {
    networkOptions = h(LoRANetwork, { metadata });
  }

  return [
    h(
      "div",
      { key: "network", className: "row space-apart" },
      h(MetaAttribute, {
        containerProps: { id: "network-module" },
        name: "Network module",
        valueClassName: "attribute",
        value: networkModule,
      }),
      h(MetaAttribute, {
        containerProps: { id: "network-type" },
        name: "Network type",
        valueClassName: "attribute",
        value: networkType,
      }),
      networkOptions,
      h(MetaAttribute, {
        name: "Network dropout",
        valueClassName: "number",
        value: metadata.get("ss_network_dropout"),
      }),

      supportsDoRA(networkType) &&
        h(MetaAttribute, {
          name: "Weight decomposition (DoRA)",
          valueClassName: "number",
          value: weightDecomposition ?? "False",
        }),
      h(MetaAttribute, {
        name: "Rank-stabilized",
        valueClassName: "number",
        value: rankStabilized ? "True" : "False",
      }),

      h(MetaAttribute, {
        name: "Gradient Checkpointing",
        valueClassName: "number",
        value: metadata.get("ss_gradient_checkpointing"),
      }),
    ),
    h(
      "div",
      { key: "network-args", className: "row space-apart" },
      h(MetaAttribute, {
        name: "Network args",
        containerProps: {
          id: "network-args",
          style: { gridColumn: "1 / span 6" },
        },
        valueClassName: "args",
        value: networkArgs ? JSON.stringify(networkArgs) : "None",
      }),
    ),
  ];
  // h("div", {}, [
  //   h("div", { title: "seed" }, metadata.get("ss_seed")),
  //   h("div", { title: "Training started at" }, trainingStart),
  //   h("div", { title: "Training ended at" }, trainingEnded),
  // ]),
}

function DiagOFTNetwork({ metadata }) {
  const [dims, setDims] = React.useState([metadata.get("ss_network_dim")]);
  React.useEffect(() => {
    trySyncMessage(
      { messageType: "dims", name: mainFilename },
      mainFilename,
    ).then((resp) => {
      setDims(resp.dims);
    });
  }, []);
  return [
    h(MetaAttribute, {
      name: "Network blocks",
      valueClassName: "rank",
      value: dims.join(", "),
    }),
  ];
}

function BOFTNetwork({ metadata }) {
  const [dims, setDims] = React.useState([metadata.get("ss_network_dim")]);
  React.useEffect(() => {
    trySyncMessage(
      { messageType: "dims", name: mainFilename },
      mainFilename,
    ).then((resp) => {
      setDims(resp.dims);
    });
  }, []);
  return [
    h(MetaAttribute, {
      name: "Network factor",
      valueClassName: "rank",
      value: dims.join(", "),
    }),
  ];
}

function LoKrNetwork({ metadata }) {
  return [
    h(MetaAttribute, {
      name: "Network Rank/Dimension",
      containerProps: { id: "network-rank" },
      valueClassName: "rank",
      value: metadata.get("ss_network_dim"),
      key: "network-rank",
    }),
    h(MetaAttribute, {
      name: "Network Alpha",
      containerProps: { id: "network-alpha" },
      valueClassName: "alpha",
      value: metadata.get("ss_network_alpha"),
      key: "network-alpha",
    }),
  ];
}

function LoRANetwork({ metadata }) {
  const [alphas, setAlphas] = React.useState([
    metadata?.get("ss_network_alpha") ?? undefined,
  ]);
  const [dims, setDims] = React.useState([
    metadata?.get("ss_network_dim") ?? undefined,
  ]);
  React.useEffect(() => {
    trySyncMessage(
      { messageType: "alphas", name: mainFilename },
      mainFilename,
    ).then((resp) => {
      setAlphas(resp.alphas);
    });
    trySyncMessage(
      { messageType: "dims", name: mainFilename },
      mainFilename,
    ).then((resp) => {
      setDims(resp.dims);
    });
  }, []);

  return [
    h(MetaAttribute, {
      name: "Network Rank/Dimension",
      containerProps: { id: "network-rank" },
      valueClassName: "rank",
      value: dims.join(", "),
      key: "network-rank",
    }),
    h(MetaAttribute, {
      name: "Network Alpha",
      containerProps: { id: "network-alpha" },
      valueClassName: "alpha",
      value: alphas
        .filter((alpha) => alpha)
        .map((alpha) => {
          if (typeof alpha === "number") {
            return alpha.toPrecision(2);
          }

          if (alpha.includes(".")) {
            return Number.parseFloat(alpha).toPrecision(2);
          }
          return Number.parseInt(alpha);
        })
        .join(", "),
      key: "network-alpha",
    }),
  ];
}

function LRScheduler({ metadata }) {
  const lrScheduler = metadata.has("ss_lr_scheduler_type")
    ? metadata.get("ss_lr_scheduler_type")
    : metadata.get("ss_lr_scheduler");

  return h(
    "div",
    { className: "row space-apart" },
    h(MetaAttribute, {
      name: "LR Scheduler",

      containerProps: { style: { gridColumn: "1 / span 3" } },
      value: lrScheduler,
    }),

    metadata.has("ss_lr_warmup_steps") &&
      h(MetaAttribute, {
        name: "Warmup steps",
        valueClassName: "lr number",
        value: metadata.get("ss_lr_warmup_steps"),
      }),
    (metadata.get("ss_unet_lr") === "None" ||
      metadata.get("ss_text_encoder_lr") === "None") &&
      h(MetaAttribute, {
        name: "Learning rate",
        valueClassName: "lr number",
        value: metadata.get("ss_learning_rate"),
      }),
    h(MetaAttribute, {
      name: "UNet learning rate",
      valueClassName: "lr number",
      value: metadata.get("ss_unet_lr"),
    }),
    h(MetaAttribute, {
      name: "Text encoder learning rate",
      valueClassName: "lr number",
      value: metadata.get("ss_text_encoder_lr"),
    }),
  );
}

function Optimizer({ metadata }) {
  return h(
    "div",
    { className: "row space-apart" },
    h(MetaAttribute, {
      name: "Optimizer",

      containerProps: { className: "span-3" },
      value: metadata.get("ss_optimizer"),
    }),
    h(MetaAttribute, {
      name: "Seed",
      valueClassName: "number",
      value: metadata.get("ss_seed"),
    }),
  );
}

function Weight({ metadata, filename }) {
  // const [precision, setPrecision] = React.useState("");
  // const [averageStrength, setAverageStrength] = React.useState(undefined);
  // const [averageMagnitude, setAverageMagnitude] = React.useState(undefined);

  // React.useEffect(() => {
  //   setAverageStrength(get_average_strength(buffer));
  //   setAverageMagnitude(get_average_magnitude(buffer));
  // }, []);

  // React.useEffect(() => {
  //   trySyncMessage(
  //     { messageType: "precision", name: mainFilename },
  //     mainFilename,
  //   ).then((resp) => {
  //     setPrecision(resp.precision);
  //   });
  // }, []);

  if (!metadata) {
    return h(Blocks, { metadata, filename });
  }

  return [
    h(
      "div",
      { className: "row space-apart", key: "norms" },
      h(MetaAttribute, {
        name: "Max Grad Norm",
        valueClassName: "number",
        value: metadata.get("ss_max_grad_norm"),
      }),
      h(MetaAttribute, {
        name: "Scale Weight Norms",
        valueClassName: "number",
        value: metadata.get("ss_scale_weight_norms"),
      }),
      h(MetaAttribute, {
        name: "CLIP Skip",
        valueClassName: "number",
        value: metadata.get("ss_clip_skip"),
      }),
      // h(MetaAttribute, {
      //   name: "Average vector magnitude, UNet + TE",
      //   valueClassName: "number",
      //   value: averageMagnitude?.toPrecision(4),
      // }),
    ),
    h(
      "div",
      { className: "row space-apart", key: "precision" },
      h(Precision),
      h(MetaAttribute, {
        name: "Mixed precision",
        valueClassName: "number",
        value: metadata.get("ss_mixed_precision"),
      }),
      metadata.has("ss_full_fp16") &&
        h(MetaAttribute, {
          name: "Full fp16",
          valueClassName: "number",
          value: metadata.get("ss_full_fp16"),
        }),
      metadata.has("ss_full_bf16") &&
        h(MetaAttribute, {
          name: "Full bf16",
          valueClassName: "number",
          value: metadata.get("ss_full_bf16"),
        }),
      h(MetaAttribute, {
        name: "fp8 base",
        valueClassName: "number",
        value: metadata.get("ss_fp8_base"),
      }),
    ),
    h(Blocks, { key: "blocks", metadata, filename }),
  ];
}

function Precision(props = {}) {
  const [precision, setPrecision] = React.useState("");

  React.useEffect(() => {
    trySyncMessage(
      { messageType: "precision", name: mainFilename },
      mainFilename,
    ).then((resp) => {
      setPrecision(resp.precision);
    });
  }, []);

  return h(MetaAttribute, {
    name: "Precision",
    valueClassName: "number",
    value: precision,
  });
}

// CHART.JS DEFAULTS
// Chart.defaults.font.size = 16;
// Chart.defaults.font.family = "monospace";

function scale_weight() {
  // get base_names
  // get scale weight
  // get progress
}

function Blocks({ filename }) {
  const [hasBlockWeights, setHasBlockWeights] = React.useState(false);
  const [teMagBlocks, setTEMagBlocks] = React.useState(new Map());
  const [unetMagBlocks, setUnetMagBlocks] = React.useState(new Map());
  const [normProgress, setNormProgress] = React.useState(0);
  const [currentCount, setCurrentCount] = React.useState(0);
  const [totalCount, setTotalCount] = React.useState(0);

  const [startTime, setStartTime] = React.useState(undefined);
  const [currentBaseName, setCurrentBaseName] = React.useState("");
  const [canHaveBlockWeights, setCanHaveBlockWeights] = React.useState(false);

  const teChartRef = React.useRef(null);
  const unetChartRef = React.useRef(null);

  React.useEffect(() => {
    if (!hasBlockWeights) {
      return;
    }

    setStartTime(performance.now());

    listenProgress("l2_norms_progress", filename).then(async (getProgress) => {
      let progress;
      while ((progress = await getProgress().next())) {
        const value = progress.value;
        setCurrentBaseName(value.baseName);
        setCurrentCount(value.currentCount);
        setTotalCount(value.totalCount);
        setNormProgress(value.currentCount / value.totalCount);
      }
    });

    trySyncMessage(
      {
        messageType: "l2_norm",
        name: filename,
        reply: true,
      },
      filename,
    ).then((resp) => {
      setTEMagBlocks(resp.norms.te);
      setUnetMagBlocks(resp.norms.unet);
    });

    return function cleanup() {};
  }, [hasBlockWeights]);

  React.useEffect(() => {
    trySyncMessage(
      {
        messageType: "network_type",
        name: filename,
        reply: true,
      },
      filename,
    ).then((resp) => {
      if (
        resp.networkType === "LoRA" ||
        resp.networkType === "LoRAFA" ||
        resp.networkType === "DyLoRA" ||
        resp.networkType === "GLoRA" ||
        // Assuming networkType of none could have block weights
        resp.networkType === undefined
      ) {
        setCanHaveBlockWeights(true);
        // trySyncMessage(
        //   {
        //     messageType: "precision",
        //     name: filename,
        //     reply: true,
        //   },
        //   filename,
        // ).then((resp) => {
        //   if (resp.precision == "bf16") {
        //     setCanHaveBlockWeights(false);
        //   }
        // });
      }
    });
  }, []);

  React.useEffect(() => {
    if (!teChartRef.current && !unetChartRef.current) {
      return;
    }

    const makeChart = (dataset, chartRef) => {
      const data = {
        // A labels array that can contain any sort of values
        labels: dataset.map(([k, _]) => k),
        // Our series array that contains series objects or in this case series data arrays
        series: [
          dataset.map(([_k, v]) => v["mean"]),
          // dataset.map(([k, v]) => strBlocks.get(k)),
        ],
      };
      // console.log("chartdata", data);
      const chart = new LineChart(chartRef.current, data, {
        chartPadding: {
          right: 60,
          top: 30,
          bottom: 30,
        },
        // seriesBarDistance: 15,
        fullWidth: true,
        axisX: {
          // showGrid: false,
          // offset: 10,
          // offset: -60,
          // position: "start",
        },
        axisY: {
          offset: 60,
          // scaleMinSpace: 100,
          // position: "end",
        },
        // plugins: [
        //   Chartist.plugins.ctPointLabels({
        //     labelOffset: {
        //       x: 10,
        //       y: -10,
        //     },
        //     textAnchor: "middle",
        //     labelInterpolationFnc: function (value) {
        //       return value.toPrecision(4);
        //     },
        //   }),
        // ],
      });

      let seq = 0;

      // Once the chart is fully created we reset the sequence
      chart.on("created", function () {
        seq = 0;
      });
      chart.on("draw", function (data) {
        if (data.type === "point") {
          // If the drawn element is a line we do a simple opacity fade in. This could also be achieved using CSS3 animations.
          data.element.animate({
            opacity: {
              // The delay when we like to start the animation
              begin: seq++ * 40,
              // Duration of the animation
              dur: 90,
              // The value where the animation should start
              from: 0,
              // The value where it should end
              to: 1,
            },
            x1: {
              begin: seq++ * 20,
              dur: 90,
              from: data.x - 20,
              to: data.x,
              // You can specify an easing function name or use easing functions from Chartist.Svg.Easing directly
              easing: easings.easeOutQuart,
            },
          });
        }
      });
    };

    if (teMagBlocks.size > 0) {
      makeChart(
        // We are removing elements that are 0 because they cause the chart to find them as undefined
        Array.from(teMagBlocks).filter(([_k, v]) => v["mean"] !== 0),
        teChartRef,
      );
    }
    if (unetMagBlocks.size > 0) {
      makeChart(
        // We are removing elements that are 0 because they cause the chart to find them as undefined
        Array.from(unetMagBlocks).filter(([_k, v]) => v["mean"] !== 0),
        unetChartRef,
      );
    }
  }, [teMagBlocks, unetMagBlocks]);

  if (!canHaveBlockWeights) {
    return h(
      "div",
      { className: "block-weights-container" },
      "Block weights not supported for this network type or precision.",
    );
  }

  if (!hasBlockWeights) {
    return h(
      "div",
      { className: "block-weights-container" },
      h(
        "button",
        {
          className: "primary",
          onClick: (e) => {
            e.preventDefault();
            setHasBlockWeights((state) => (state ? false : true));
          },
        },
        "Get block weights",
      ),
    );
  }

  let teBlockWeights = [];

  if (teMagBlocks.size > 0) {
    teBlockWeights = [
      h(
        "h3",
        { key: "text-encoder-block-weights-header" },
        "Text encoder block weights",
      ),
      h("div", {
        key: "text-encoder-chart",
        ref: teChartRef,
        className: "chart",
      }),
      h(
        "div",
        {
          key: "text-encoder-block-weights",
          className: "block-weights text-encoder",
        },
        Array.from(teMagBlocks)
          .sort(([a, _], [b, _v]) => a > b)
          .map(([k, v]) => {
            return h(
              "div",
              { key: k },
              // h(MetaAttribute, {
              //   name: `${k} average strength`,
              //   value: teStrBlocks[k].toPrecision(6),
              //   valueClassName: "number",
              // }),
              h(MetaAttribute, {
                className: "te-block",
                name: `${k} avg l2 norm ${v["metadata"]["type"]}`,
                value: v["mean"].toPrecision(6),
                valueClassName: "number",
              }),
            );
          }),
      ),
    ];
  }

  let unetBlockWeights = [];
  if (unetMagBlocks.size > 0) {
    unetBlockWeights = [
      h("h3", { key: "unet-header" }, "UNet block weights"),
      h("div", { key: "unet-chart", ref: unetChartRef, className: "chart" }),
      h(
        "div",
        { key: "unet-block-weights", className: "block-weights unet" },
        Array.from(unetMagBlocks).map(([k, v]) => {
          return h(
            "div",
            { key: k },
            // h(MetaAttribute, {
            //   name: `${k} average strength`,
            //   value: v.toPrecision(6),
            //   valueClassName: "number",
            // }),
            h(MetaAttribute, {
              className: "unet-block",
              name: `${k} avg l2 norm  ${v["metadata"]["type"]}`,
              value: v["mean"].toPrecision(6),
              valueClassName: "number",
            }),
          );
        }),
      ),
    ];
  }

  if (
    unetBlockWeights.length === 0 &&
    teBlockWeights.length === 0 &&
    hasBlockWeights === true
  ) {
    const elapsedTime = performance.now() - startTime;
    const remaining =
      (elapsedTime * totalCount) / normProgress - elapsedTime * totalCount;
    const perSecond = currentCount / (elapsedTime / 1_000);

    return h(
      "div",
      { className: "block-weights-container" },
      // { className: "marquee" },
      h(
        "span",
        null,
        `Loading block weights... ${(normProgress * 100).toFixed(
          2,
        )}% ${currentCount}/${totalCount} ${perSecond.toFixed(2)}it/s ${(
          remaining / 1_000_000
        ).toFixed(2)}s remaining ${currentBaseName} `,
      ),
    );
  }

  return h("div", { className: "block-weights-container" }, [
    teBlockWeights,
    unetBlockWeights,
  ]);
}

function EpochStep({ metadata }) {
  return h(
    "div",
    { className: "row space-apart" },
    h(MetaAttribute, {
      name: "Epoch",
      valueClassName: "number",
      value: metadata.get("ss_epoch"),
    }),
    h(MetaAttribute, {
      name: "Steps",
      valueClassName: "number",
      value: metadata.get("ss_steps"),
    }),
    metadata.get("ss_max_train_steps") === undefined &&
      h(MetaAttribute, {
        name: "Max Train Epochs",
        valueClassName: "number",
        value: metadata.get("ss_max_train_epochs"),
      }),
    metadata.get("ss_max_train_epochs") === undefined &&
      h(MetaAttribute, {
        name: "Max Train Steps",
        valueClassName: "number",
        value: metadata.get("ss_max_train_steps"),
      }),
  );
}
function Batch({ metadata }) {
  let batchSize;
  if (metadata.has("ss_batch_size_per_device")) {
    batchSize = metadata.get("ss_batch_size_per_device");
  } else {
    // The batch size is found inside the datasets.
    if (metadata.has("ss_datasets")) {
      const datasets = JSON.parse(metadata.get("ss_datasets"));

      for (const dataset of datasets) {
        if ("batch_size_per_device" in dataset) {
          batchSize = dataset["batch_size_per_device"];
        }
      }
    }
  }

  return h(
    "div",
    { className: "row space-apart" },
    h(MetaAttribute, {
      name: "Num train images",
      valueClassName: "number",
      value: metadata.get("ss_num_train_images"),
    }),
    h(MetaAttribute, {
      name: "Num batches per epoch",
      valueClassName: "number",
      value: metadata.get("ss_num_batches_per_epoch"),
    }),
    h(MetaAttribute, {
      name: "Batch size",
      valueClassName: "number",
      value: batchSize,
    }),
    h(MetaAttribute, {
      name: "Gradient Accumulation Steps",
      valueClassName: "number",
      value: metadata.get("ss_gradient_accumulation_steps"),
    }),
  );
}

function Noise({ metadata }) {
  return h(
    "div",
    { className: "row space-apart" },
    h(MetaAttribute, {
      name: "IP noise gamma",
      valueClassName: "number",
      value: metadata.get("ss_ip_noise_gamma"),
      ...(metadata.get("ss_ip_noise_gamma_random_strength") != undefined && {
        secondaryName: "Random strength:",
        secondary:
          metadata.get("ss_ip_noise_gamma_random_strength") == "True"
            ? "True"
            : "False",
        // secondaryClassName: "number",
      }),
    }),
    h(MetaAttribute, {
      name: "Noise offset",
      valueClassName: "number",
      value: metadata.get("ss_noise_offset"),
      ...(metadata.get("ss_ip_noise_gamma_random_strength") != undefined && {
        secondaryName: "Random strength:",
        secondary:
          metadata.get("ss_noise_offset_random_strength") == "True"
            ? "True"
            : "False",
        // secondaryClassName: "number",
      }),
    }),
    h(MetaAttribute, {
      name: "Adaptive noise scale",
      valueClassName: "number",
      value: metadata.get("ss_adaptive_noise_scale"),
    }),
    h(MultiresNoise, { metadata }),
  );
}

function MultiresNoise({ metadata }) {
  if (metadata.get("ss_multires_noise_iterations") === "None") {
    return [];
  }

  return [
    h(MetaAttribute, {
      name: "MultiRes Noise Iterations",
      valueClassName: "number",
      value: metadata.get("ss_multires_noise_iterations"),
      key: "multires noise iterations",
    }),
    h(MetaAttribute, {
      name: "MultiRes Noise Discount",
      valueClassName: "number",
      value: metadata.get("ss_multires_noise_discount"),
      key: "multires noise discount",
    }),
  ];
}

function Loss({ metadata }) {
  return h(
    "div",
    { className: "row space-apart" },
    h(MetaAttribute, {
      name: "Loss type",
      valueClassName: "attribute",
      value: metadata.get("ss_loss_type"),
    }),
    metadata.get("ss_loss_type") === "huber" &&
      h(MetaAttribute, {
        name: "Huber schedule",
        valueClassName: "attribute",
        value: metadata.get("ss_huber_schedule"),
      }),
    metadata.get("ss_loss_type") === "huber" &&
      h(MetaAttribute, {
        name: "Huber c",
        valueClassName: "number",
        value: metadata.get("ss_huber_c"),
      }),
    h(MetaAttribute, {
      name: "Debiased Estimation",
      valueClassName: "boolean",
      value: metadata.get("ss_debiased_estimation") ?? "False",
    }),
    h(MetaAttribute, {
      name: "Min SNR Gamma",
      valueClassName: "number",
      value: metadata.get("ss_min_snr_gamma"),
    }),
    h(MetaAttribute, {
      name: "Zero Terminal SNR",
      valueClassName: "boolean",
      value: metadata.get("ss_zero_terminal_snr"),
    }),
    metadata.has("ss_masked_loss") != undefined &&
      h(MetaAttribute, {
        name: "Masked Loss",
        value: metadata.get("ss_masked_loss"),
      }),
  );
}

function CaptionDropout({ metadata }) {
  return h(
    "div",
    { className: "row space-apart" },
    h(MetaAttribute, {
      name: "Max token length",
      valueClassName: "number",
      value: metadata.get("ss_max_token_length"),
    }),

    h(MetaAttribute, {
      name: "Caption dropout rate",
      valueClassName: "number",
      value: metadata.get("ss_caption_dropout_rate"),
    }),
    h(MetaAttribute, {
      name: "Caption dropout every n epochs",
      valueClassName: "number",
      value: metadata.get("ss_caption_dropout_every_n_epochs"),
    }),

    h(MetaAttribute, {
      name: "Caption tag dropout rate",
      valueClassName: "number",
      value: metadata.get("ss_caption_tag_dropout_rate"),
    }),
  );
}

function Dataset({ metadata }) {
  let datasets;
  if (metadata.has("ss_datasets")) {
    datasets = JSON.parse(metadata.get("ss_datasets"));
  } else {
    datasets = [];
  }
  return h(
    "div",
    null,
    h("h2", null, "Dataset"),

    datasets.map((dataset, i) => {
      return h(Buckets, { key: i, dataset, metadata });
    }),
  );
}

function Buckets({ dataset, metadata }) {
  return [
    h(
      "div",
      { key: "buckets", className: "row space-apart" },
      h(MetaAttribute, {
        name: "Buckets",
        value: dataset["enable_bucket"] ? "True" : "False",
      }),
      h(MetaAttribute, {
        name: "Min bucket resolution",
        valueClassName: "number",
        value: dataset["min_bucket_reso"],
      }),
      h(MetaAttribute, {
        name: "Max bucket resolution",
        valueClassName: "number",
        value: dataset["max_bucket_reso"],
      }),
      h(MetaAttribute, {
        name: "Resolution",
        valueClassName: "number",
        value: `${dataset["resolution"][0]}x${dataset["resolution"][0]}`,
      }),
    ),

    h("div", { key: "bucket-info" }, h(BucketInfo, { metadata, dataset })),
    h(
      "h3",
      { key: "subsets-header", className: "row space-apart" },
      "Subsets:",
    ),
    h(
      "div",
      { key: "subsets", className: "subsets" },
      dataset["subsets"].map((subset, i) =>
        h(Subset, {
          key: `subset-${subset["image_dir"]}-${i}`,
          metadata,
          subset,
        }),
      ),
    ),
    h("h3", { key: "header-tag-frequencies" }, "Tag frequencies"),
    h(
      "div",
      { key: "tag-frequencies", className: "tag-frequencies row space-apart" },
      Object.entries(dataset["tag_frequency"]).map(([dir, frequency]) =>
        h(
          "div",
          { key: dir },
          h("h3", {}, dir),
          h(TagFrequency, {
            key: "tag-frequency",
            tagFrequency: frequency,
            metadata,
          }),
        ),
      ),
    ),
  ];
}

function BucketInfo({ metadata, dataset }) {
  // No bucket info
  if (!dataset["bucket_info"]) {
    return;
  }

  // No buckets data
  if (!dataset["bucket_info"]["buckets"]) {
    return;
  }

  return h("div", { className: "bucket-infos" }, [
    Object.entries(dataset["bucket_info"]["buckets"]).map(([key, bucket]) => {
      return h(
        "div",
        { key, className: "bucket" },
        h(MetaAttribute, {
          name: `Bucket ${key}`,
          value: `${bucket["resolution"][0]}x${bucket["resolution"][1]}: ${
            bucket["count"]
          } image${bucket["count"] > 1 ? "s" : ""}`,
        }),
      );
    }),
  ]);
}

function Subset({ subset, metadata }) {
  const tf = (v, defaults = undefined, opts) => {
    let className = "";
    if (v === true) {
      if (v !== defaults) {
        className = "changed";
      }
      return {
        valueClassName: opts?.valueClassName ?? "" + " option " + className,
        value: "true",
      };
    }
    if (v !== defaults) {
      className = "changed";
    }
    return {
      valueClassName: opts?.valueClassName ?? "" + " option " + className,
      value: "false",
    };
  };

  return h(
    "div",
    { className: "subset row space-apart" },
    h(MetaAttribute, {
      name: "Image count",
      value: subset["img_count"],
      valueClassName: "number",
    }),
    h(MetaAttribute, {
      name: "Image dir",
      value: subset["image_dir"],
      valueClassName: "",
    }),
    h(MetaAttribute, {
      name: "Flip augmentation",
      ...tf(subset["flip_aug"], false),
    }),
    h(MetaAttribute, {
      name: "Color augmentation",
      ...tf(subset["color_aug"], false),
    }),
    h(MetaAttribute, {
      name: "Num repeats",
      value: subset["num_repeats"],
      valueClassName: "number",
    }),
    h(MetaAttribute, {
      name: "Is regularization",
      ...tf(subset["is_reg"], false),
    }),
    h(MetaAttribute, { name: "Class token", value: subset["class_tokens"] }),
    h(MetaAttribute, {
      name: "Keep tokens",
      value: subset["keep_tokens"],
      valueClassName: "number",
    }),
    "keep_tokens_separator" in subset &&
      h(MetaAttribute, {
        name: "Keep tokens separator",
        value: subset["keep_tokens_separator"],
      }),
    "caption_separator" in subset &&
      h(MetaAttribute, {
        name: "Caption separator",
        value: subset["caption_separator"],
      }),
    "secondary_separator" in subset &&
      h(MetaAttribute, {
        name: "Secondary separator",
        value: subset["secondary_separator"],
      }),
    "enable_wildcard" in subset &&
      h(MetaAttribute, {
        name: "Enable wildcard",
        ...tf(subset["enable_wildcard"], false),
      }),
    "shuffle_caption" in subset &&
      h(MetaAttribute, {
        name: "Shuffle caption",
        ...tf(subset["shuffle_caption"], false),
      }),
    "caption_prefix" in subset &&
      h(MetaAttribute, {
        name: "Caption prefix",
        value: subset["caption_prefix"],
      }),
    "caption_suffix" in subset &&
      h(MetaAttribute, {
        name: "Caption suffix",
        value: subset["caption_suffix"],
      }),
  );
}

function TagFrequency({ tagFrequency, metadata }) {
  const [showMore, setShowMore] = React.useState(false);

  const allTags = Object.entries(tagFrequency).sort((a, b) => a[1] < b[1]);
  const sortedTags = showMore == false ? allTags.slice(0, 50) : allTags;

  return [
    sortedTags.map(([tag, count], i) => {
      const alt = i % 2 > 0 ? " alt-row" : "";
      return h(
        "div",
        { className: "tag-frequency" + alt, key: tag },
        h("div", {}, count),
        h("div", {}, tag),
      );
    }),
    h(
      "div",
      { key: "show-more" },
      showMore === false && allTags.length > sortedTags.length
        ? h(
            "button",
            {
              onClick: () => {
                setShowMore(true);
              },
            },
            "Show more",
          )
        : showMore === true &&
            h(
              "button",
              {
                onClick: () => {
                  setShowMore(false);
                },
              },
              "Show less",
            ),
    ),
  ];
}

function Advanced({ metadata, filename }) {
  const [baseNames, setBaseNames] = React.useState([]);
  const [showBaseNames, setShowBlockNames] = React.useState(false);

  const [textEncoderKeys, setTextEncoderKeys] = React.useState([]);
  const [showTextEncoderKeys, setShowTextEncoderKeys] = React.useState(false);

  const [unetKeys, setUnetKeys] = React.useState([]);
  const [showUnetKeys, setShowUnetKeys] = React.useState(false);

  const [allKeys, setAllKeys] = React.useState([]);
  const [showAllKeys, setShowAllKeys] = React.useState(false);

  const [canHaveStatistics, setCanHaveStatistics] = React.useState(false);

  const advancedRef = React.createRef();

  React.useEffect(() => {
    trySyncMessage(
      { messageType: "base_names", name: filename },
      filename,
    ).then((resp) => {
      resp.baseNames.sort();
      setBaseNames(resp.baseNames);
    });

    trySyncMessage(
      { messageType: "text_encoder_keys", name: filename },
      filename,
    ).then((resp) => {
      resp.textEncoderKeys.sort();
      // console.log("text encoder keys", resp.textEncoderKeys);
      setTextEncoderKeys(resp.textEncoderKeys);
    });

    trySyncMessage({ messageType: "unet_keys", name: filename }, filename).then(
      (resp) => {
        resp.unetKeys.sort();
        // console.log("unet keys", resp.unetKeys);
        setUnetKeys(resp.unetKeys);
      },
    );

    trySyncMessage({ messageType: "keys", name: filename }, filename).then(
      (resp) => {
        resp.keys.sort();

        setAllKeys(resp.keys);
      },
    );
  }, []);

  React.useEffect(() => {
    trySyncMessage(
      {
        messageType: "network_type",
        name: filename,
        reply: true,
      },
      filename,
    ).then((resp) => {
      if (
        resp.networkType === "LoRA" ||
        resp.networkType === "LoRAFA" ||
        resp.networkType === "DyLoRA" ||
        // Assuming networkType of none could have block weights
        resp.networkType === undefined
      ) {
        setCanHaveStatistics(true);
        trySyncMessage(
          {
            messageType: "precision",
            name: filename,
            reply: true,
          },
          filename,
        ).then((resp) => {
          if (resp.precision == "bf16") {
            setCanHaveStatistics(false);
          }
        });
      }
    });
  }, []);

  if (DEBUG) {
    React.useEffect(() => {
      advancedRef.current.scrollIntoView({ behavior: "smooth" });
    }, []);
  }

  return [
    h(
      "h2",
      { key: "header-advanced", id: "advanced", ref: advancedRef },
      "Advanced",
    ),
    h("div", { key: "advanced", className: "row" }, [
      h(
        "div",
        { key: "base name keys" },
        showBaseNames
          ? h(BaseNames, { baseNames })
          : h("div", null, `Base name keys: ${baseNames.length}`),
      ),
      h(
        "div",
        { key: "text encoder name keys" },
        showTextEncoderKeys
          ? h(TextEncoderKeys, { textEncoderKeys })
          : h("div", null, `Text encoder keys: ${textEncoderKeys.length}`),
      ),
      h(
        "div",
        { key: "unet keys" },
        showUnetKeys
          ? h(UnetKeys, { unetKeys })
          : h("div", null, `Unet keys: ${unetKeys.length}`),
      ),
      h(
        "div",
        { key: "all keys" },
        showAllKeys
          ? h(AllKeys, { allKeys })
          : h("div", null, `All keys: ${allKeys.length}`),
      ),
    ]),
    !canHaveStatistics
      ? h(BaseNames, { key: "base-names", baseNames })
      : h(Statistics, { key: "statistics", baseNames, filename }),
  ];
}

const DEBUG = new URLSearchParams(document.location.search).has("DEBUG");

function Statistics({ baseNames, filename }) {
  const [calcStatistics, setCalcStatistics] = React.useState(false);
  const [hasStatistics, setHasStatistics] = React.useState(false);
  const [bases, setBases] = React.useState([]);
  const [statisticProgress, setStatisticProgress] = React.useState(0);
  const [currentCount, setCurrentCount] = React.useState(0);
  const [totalCount, setTotalCount] = React.useState(0);

  const [startTime, setStartTime] = React.useState(undefined);
  const [currentBaseName, setCurrentBaseName] = React.useState("");

  // const [scaleWeightProgress, setScaleWeightProgress] = React.useState(0);
  // const [currentScaleWeightCount, setCurrentScaleWeightCount] =
  //   React.useState(0);
  // const [totalScaleWeightCount, setTotalScaleWeightCount] = React.useState(0);

  React.useEffect(() => {
    if (!calcStatistics) {
      return;
    }

    if (baseNames.length === 0) {
      return;
    }

    console.time("scale weights");
    // trySyncMessage(
    //   {
    //     messageType: "scale_weights_with_progress",
    //     name: filename,
    //     reply: true,
    //   },
    //   filename,
    // ).then(() => {
    console.timeEnd("scale weights");
    console.log("Calculating statistics...");
    console.time("get statistics");

    setStartTime(performance.now());

    // listenProgress("statistics_progress", filename)
    //   .then(async (getProgress) => {
    //     let progress;
    //     while ((progress = await getProgress().next())) {
    //       const value = progress.value;
    //       setCurrentBaseName(value.baseName);
    //       setCurrentCount(value.currentCount);
    //       setTotalCount(value.totalCount);
    //       setNormProgress(value.currentCount / value.totalCount);
    //     }
    //   })
    //   .then(() => {
    let progress = 0;
    Promise.allSettled(
      baseNames.map(async (baseName) => {
        return trySyncMessage(
          { messageType: "norms", name: filename, baseName },
          filename,
          ["messageType", "baseName"],
        ).then((resp) => {
          progress += 1;
          // console.log("progress", progress / baseNames.length, resp);

          setCurrentBaseName(resp.baseName);
          setCurrentCount(progress);
          setTotalCount(baseNames.length);
          setStatisticProgress(progress / baseNames.length);

          return { baseName: resp.baseName, stat: resp.norms };
        });
      }),
    ).then((results) => {
      progress = 0;
      const bases = results
        .filter((v) => v.status === "fulfilled")
        .map((v) => v.value);
      setBases(bases);
      setHasStatistics(true);
      console.timeEnd("get statistics");
    });
    // });
    // });
  }, [calcStatistics, baseNames]);

  React.useEffect(() => {
    if (!calcStatistics) {
      return;
    }

    if (baseNames.length === 0) {
      return;
    }

    setStartTime(performance.now());

    listenProgress("scale_weight_progress", filename).then(
      async (getProgress) => {
        let progress;
        while ((progress = await getProgress().next())) {
          const value = progress.value;
          if (!value) {
            break;
          }
          setCurrentBaseName(value.baseName);
          setCurrentScaleWeightCount(value.currentCount);
          setTotalScaleWeightCount(value.totalCount);
          setScaleWeightProgress(value.currentCount / value.totalCount);
        }
      },
    );

    return function cleanup() {};
  }, [calcStatistics]);

  if (!hasStatistics && !calcStatistics) {
    return h(
      "div",
      null,
      h(
        "button",
        {
          onClick: (e) => {
            e.preventDefault();
            setCalcStatistics(true);
          },
        },
        "Calculate statistics",
      ),
    );
  }

  if (calcStatistics && !hasStatistics) {
    const elapsedTime = performance.now() - startTime;
    const remaining =
      (elapsedTime * totalCount) / statisticProgress -
        elapsedTime * totalCount || 0;
    const perSecond = currentCount / (elapsedTime / 1_000);

    // if (currentCount === 0) {
    //   const elapsedTime = performance.now() - startTime;
    //   const perSecond = currentScaleWeightCount / (elapsedTime / 1_000);
    //
    //   if (scaleWeightProgress === 0) {
    //     return h(
    //       "div",
    //       { className: "block-weights-container" },
    //       "Waiting for worker, please wait...",
    //     );
    //   }
    //
    //   const remaining =
    //     (elapsedTime * totalScaleWeightCount) / scaleWeightProgress -
    //     elapsedTime * totalScaleWeightCount;
    //   return h(
    //     "div",
    //     { className: "block-weights-container" },
    //     h(
    //       "span",
    //       null,
    //       `Scaling weights... ${(scaleWeightProgress * 100).toFixed(
    //         2,
    //       )}% ${currentScaleWeightCount}/${totalScaleWeightCount} ${perSecond.toFixed(
    //         2,
    //       )}it/s ${(remaining / 1_000_000).toFixed(
    //         2,
    //       )}s remaining ${currentBaseName} `,
    //     ),
    //   );
    // }

    return h(
      "div",
      { className: "block-weights-container" },
      // { className: "marquee" },
      h(
        "span",
        null,
        `Loading statistics... ${(statisticProgress * 100).toFixed(
          2,
        )}% ${currentCount}/${totalCount} ${perSecond.toFixed(2)}it/s ${(
          remaining / 1_000_000
        ).toFixed(2)}s remaining ${currentBaseName} `,
      ),
    );
  }

  const teLayers = compileTextEncoderLayers(bases);
  const unetLayers = compileUnetLayers(bases);

  return [
    DEBUG &&
      h(
        "div",
        {
          style: {
            display: "grid",
            justifyContent: "flex-end",
          },
        },
        h(
          "button",
          {
            onClick: () => {
              console.log("teLayers", teLayers);
              console.log("unetLayers", unetLayers);
              console.log(
                "bases",
                bases.map((v) => ({
                  ...v,
                  stat: Object.fromEntries(v["stat"]),
                })),
              );
            },
          },
          "debug stats",
        ),
      ),
    h("table", null, [
      h("thead", null, [
        h("th", null, "base name"),
        h("th", null, "l1 norm"),
        h("th", null, "l2 norm"),
        h("th", null, "matrix norm"),
        h("th", null, "min"),
        h("th", null, "max"),
        h("th", null, "median"),
        h("th", null, "std_dev"),
      ]),
      bases.map((base) => {
        return h(StatisticRow, {
          baseName: base.baseName,
          l1Norm: base.stat?.get("l1_norm"),
          l2Norm: base.stat?.get("l2_norm"),
          matrixNorm: base.stat?.get("matrix_norm"),
          min: base.stat?.get("min"),
          max: base.stat?.get("max"),
          median: base.stat?.get("median"),
          stdDev: base.stat?.get("std_dev"),
        });
      }),
    ]),

    teLayers.length > 0 && [
      h("div", null, h("h2", null, "Text Encoder Architecture")),
      h(
        "div",
        { id: "te-architecture" },
        h(TEArchitecture, { layers: teLayers }),
      ),
    ],
    h("div", null, h("h2", null, "UNet Architecture")),
    h(
      "div",
      { id: "unet-architecture" },
      h(UNetArchitecture, { layers: unetLayers }),
    ),
  ];
}

function compileTextEncoderLayers(bases) {
  // we have a list of names and we want to extract the different components and put back together to use
  // with Attention

  const re =
    /lora_te_text_model_encoder_layers_(?<layer_id>\d+)_(?<layer_type>mlp|self_attn)_(?<sub_type>k_proj|q_proj|v_proj|out_proj|fc1|fc2)/;

  const layers = [];

  for (const i in bases) {
    const base = bases[i];

    const match = base.baseName.match(re);

    if (match) {
      const layerId = match.groups.layer_id;
      const layerType = match.groups.layer_type;
      const subType = match.groups.sub_type;

      const layerKey = layerType === "self_attn" ? "attn" : "mlp";
      let value;
      let subKey;

      switch (subType) {
        case "k_proj":
          subKey = "k";
          break;

        case "q_proj":
          subKey = "q";
          break;

        case "v_proj":
          subKey = "v";
          break;

        case "out_proj":
          subKey = "out";
          break;

        case "fc1":
          subKey = "fc1";
          break;

        case "fc2":
          subKey = "fc2";
          break;
      }

      if (!layers[layerId]) {
        layers[layerId] = {
          [layerKey]: {
            [subKey]: base.stat?.get("l2_norm"),
          },
        };
      } else {
        if (!layers[layerId][layerKey]) {
          layers[layerId][layerKey] = {};
        }
        layers[layerId][layerKey][subKey] = base.stat?.get("l2_norm");
      }
    }
  }

  return layers;
}

const SDRE =
  /.*(?<block_type>up|down|mid)_blocks?_.*(?<block_id>\d+).*(?<type>resnets|attentions|upsamplers|downsamplers)_(?<subblock_id>\d+).*/;

const MID_SDRE =
  /.*(?<block_type>up|down|mid)_block_.*(?<type>resnets|attentions|upsamplers|downsamplers)_(?<block_id>\d+)_.*(?<subblock_id>\d+)?.*/;
const TE_SDRE = /(?<block_id>\d+).*(?<block_type>self_attn|mlp)/;
const NUM_OF_BLOCKS = 12;

function parseSDKey(key) {
  let blockIdx = -1;
  let idx;

  let isConv = false;
  let isAttention = false;
  let isSampler = false;
  let isProjection = false;
  let isFeedForward = false;

  let type;
  let blockType;
  let blockId;
  let subBlockId;
  let name;

  // Handle the text encoder
  if (key.includes("te_text_model")) {
    const matches = key.match(TE_SDRE);
    if (matches) {
      const groups = matches.groups;
      type = "encoder";
      blockId = parseInt(groups["block_id"]);
      blockType = groups["block_type"];

      name = `TE${padTwo(blockId)}`;

      if (blockType === "self_attn") {
        isAttention = true;
      }
    }
    // Handling the UNet values
  } else {
    const matches = key.match(SDRE);
    if (matches) {
      const groups = matches.groups;

      type = groups["type"];
      blockType = groups["block_type"];
      blockId = parseInt(groups["block_id"]);
      subBlockId = parseInt(groups["subblock_id"]);

      // console.log(groups["block_id"]);

      if (groups["type"] === "attentions") {
        idx = 3 * blockId + subBlockId;
        isAttention = true;
      } else if (groups["type"] === "resnets") {
        idx = 3 * blockId + subBlockId;
        isConv = true;
      } else if (
        groups["type"] === "upsamplers" ||
        groups["type"] === "downsamplers"
      ) {
        idx = 3 * blockId + 2;
        isSampler = true;
      }

      if (groups["block_type"] === "down") {
        blockIdx = 1 + idx;
        name = `IN${padTwo(idx)}`;
      } else if (groups["block_type"] === "up") {
        blockIdx = NUM_OF_BLOCKS + 1 + idx;
        name = `OUT${padTwo(idx)}`;
      } else if (groups["block_type"] === "mid") {
        blockIdx = NUM_OF_BLOCKS;
      }
      // Handle the mid block
    } else if (key.includes("mid_block_")) {
      const midMatch = key.match(MID_SDRE);
      name = `MID`;

      if (midMatch) {
        const groups = midMatch.groups;

        type = groups["type"];
        blockType = groups["block_type"];
        blockId = parseInt(groups["block_id"]);
        subBlockId = parseInt(groups["subblock_id"]);

        name = `MID${padTwo(blockId)}`;

        if (groups.type == "attentions") {
          isAttention = true;
        } else if (groups.type === "resnets") {
          isConv = true;
        }
      }

      blockIdx = NUM_OF_BLOCKS;
    }
  }

  return {
    // Used in commmon format IN01
    idx,
    // Block index between 1 and 24
    blockIdx,
    // Common name IN01
    name,
    // name of the block up, down, mid
    // id of the block (up_0, down_1)
    blockId,
    // id of the subblock (resnet, attentions)
    subBlockId,
    // resnets, attentions, upscalers, downscalers
    type,
    //
    blockType,
    // is a convolution key
    isConv,
    // is an attention key
    isAttention,
    // is a upscaler/downscaler
    isSampler,
    key,
  };
}

function padTwo(number, padWith = "0") {
  if (number < 10) {
    return `${padWith}${number}`;
  }

  return `${number}`;
}

function compileUnetLayers(bases) {
  // we have a list of names and we want to extract the different components and put back together to use
  // with Attention

  const re =
    /lora_unet_(down_blocks|mid_block|up_blocks)_(?<block_id>\d+)_(?<layer_type>mlp|self_attn)_(?<sub_type>k_proj|q_proj|v_proj|out_proj|fc1|fc2)/;

  const layers = {
    down: {},
    // { "00": layer }
    mid: {},
    up: {},
  };

  const ensureLayer = (layer, id) => {
    if (!layer[id]) {
      layer[id] = {
        proj_in: undefined,
        attn1: { k: undefined, q: undefined, v: undefined, out: undefined },
        attn2: { k: undefined, q: undefined, v: undefined, out: undefined },
        ff1: undefined,
        ff2: undefined,
        proj_out: undefined,
      };
    }

    return layer[id];
  };

  for (const i in bases) {
    const base = bases[i];

    let layer;

    if (base.baseName.includes("down")) {
      layer = layers.down;
    } else if (base.baseName.includes("up")) {
      layer = layers.up;
    } else if (base.baseName.includes("mid")) {
      layer = layers.mid;
    } else {
      continue;
    }

    let parsedKey = parseSDKey(base.baseName);

    // TODO need layer id
    layer = ensureLayer(layer, parsedKey.name);

    if (parsedKey.isAttention) {
      if (base.baseName.includes("attn1")) {
        if (base.baseName.includes("to_q")) {
          layer["attn1"]["q"] = base.stat?.get("l2_norm");
        } else if (base.baseName.includes("to_k")) {
          layer["attn1"]["k"] = base.stat?.get("l2_norm");
        } else if (base.baseName.includes("to_v")) {
          layer["attn1"]["v"] = base.stat?.get("l2_norm");
        } else if (base.baseName.includes("to_out")) {
          layer["attn1"]["out"] = base.stat?.get("l2_norm");
        }
      } else if (base.baseName.includes("attn2")) {
        if (base.baseName.includes("to_q")) {
          layer["attn2"]["q"] = base.stat?.get("l2_norm");
        } else if (base.baseName.includes("to_k")) {
          layer["attn2"]["k"] = base.stat?.get("l2_norm");
        } else if (base.baseName.includes("to_v")) {
          layer["attn2"]["v"] = base.stat?.get("l2_norm");
        } else if (base.baseName.includes("to_out")) {
          layer["attn2"]["out"] = base.stat?.get("l2_norm");
        }
      } else if (base.baseName.includes("ff_net_0_proj")) {
        layer["ff1"] = base.stat?.get("l2_norm");
      } else if (base.baseName.includes("ff_net_2")) {
        layer["ff2"] = base.stat?.get("l2_norm");
      } else if (base.baseName.includes("proj_in")) {
        layer["proj_in"] = base.stat?.get("l2_norm");
      } else if (base.baseName.includes("proj_out")) {
        layer["proj_out"] = base.stat?.get("l2_norm");
      }
    } else if (parsedKey.isConv) {
      if (base.baseName.includes("conv1")) {
        layer["conv1"] = base.stat?.get("l2_norm");
      } else if (base.baseName.includes("time_emb_proj")) {
        layer["time_emb_proj"] = base.stat?.get("l2_norm");
      } else if (base.baseName.includes("conv2")) {
        layer["conv2"] = base.stat?.get("l2_norm");
      } else if (base.baseName.includes("conv_shortcut")) {
        layer["conv_shortcut"] = base.stat?.get("l2_norm");
      }
    } else if (parsedKey.isSampler) {
      if (base.baseName.includes("conv")) {
        layer["conv"] = base.stat?.get("l2_norm");
      }
    }
  }

  return layers;
}

function StatisticRow({
  baseName,
  l1Norm,
  l2Norm,
  matrixNorm,
  min,
  max,
  median,
  stdDev,
}) {
  return h(
    "tr",
    null,
    h("td", null, baseName),
    h("td", null, l1Norm?.toPrecision(4)),
    h("td", null, l2Norm?.toPrecision(4)),
    h("td", null, matrixNorm?.toPrecision(4)),
    h("td", null, min?.toPrecision(4)),
    h("td", null, max?.toPrecision(4)),
    h("td", null, median?.toPrecision(4)),
    h("td", null, stdDev?.toPrecision(4)),
  );
}

function UnetKeys({ unetKeys }) {
  return [
    h("h3", { key: "unet-keys-header" }, "UNet keys"),
    h(
      "ul",
      { key: "unet-keys" },
      unetKeys.map((unetKey) => {
        return h("li", { key: unetKey }, unetKey);
      }),
    ),
  ];
}
function TextEncoderKeys({ textEncoderKeys }) {
  return [
    h("h3", { key: "text-encoder-keys-header" }, "Text encoder keys"),
    h(
      "ul",
      { key: "text-encoder-keys" },
      textEncoderKeys.map((textEncoderKey) => {
        return h("li", { key: textEncoderKey }, textEncoderKey);
      }),
    ),
  ];
}

function BaseNames({ baseNames }) {
  return [
    h("h3", { key: "header-base-names" }, "Base names"),
    h(
      "ul",
      { key: "base-names" },
      baseNames.map((baseName) => {
        return h("li", { key: baseName }, baseName);
      }),
    ),
  ];
}

function AllKeys({ allkeys }) {
  return [
    h("h3", { key: "all-keys-header" }, "All keys"),
    h(
      "ul",
      { key: "all-keys" },
      allKeys.map((key) => {
        return h("li", { key }, key);
      }),
    ),
  ];
}

// layers = { self_attn: { k: number, q: number, v: number, out: number }, mlp: { fc1, fc2 }}[]
function TEArchitecture({ layers }) {
  // 0-11 layers of attention
  //
  // (transformer): Transformer(
  //   (resblocks): ModuleList(
  //     (0-11): 12 x ResidualAttentionBlock(
  //       (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  //       (attn): MultiheadAttention(
  //         (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
  //       )
  //       (ls_1): Identity()
  //       (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  //       (mlp): Sequential(
  //         (c_fc): Linear(in_features=768, out_features=3072, bias=True)
  //         (gelu): QuickGELU()
  //         (c_proj): Linear(in_features=3072, out_features=768, bias=True)
  //       )
  //       (ls_2): Identity()
  //     )
  //   )
  // )
  return layers.map(({ attn, mlp }, i) => {
    return h(
      "div",
      { className: "text-encoder-layer-block" },
      h("h3", null, `layer ${i + 1}`),
      h(
        "svg",
        { className: "attention-layer", width: "15.5em", height: "420" },
        h(
          "defs",
          null,
          h(
            "marker",
            {
              id: "head",
              orient: "auto",
              markerWidth: 3,
              markerHeight: 4,
              refX: "0.1",
              refY: "2",
            },
            h("path", { d: "M0,0 V4 L2,2 Z", fill: "currentColor" }),
          ),
          // h(
          //   "filter",
          //   { id: "filter1", x: 0, y: 0 },
          //   h("feOffset", {
          //     result: "offOut",
          //     in: "SourceAlpha",
          //     dx: 1,
          //     dy: 1,
          //   }),
          //   h("feGaussianBlur", {
          //     result: "blurOut",
          //     in: "offOut",
          //     stdDeviation: 1,
          //   }),
          //   h("feBlend", {
          //     in: "SourceGraphic",
          //     in2: "blurOut",
          //     mode: "normal",
          //   }),
          // ),
        ),
        h(SimpleWeight, {
          groupProps: { transform: "translate(10, 25)" },
          titleProps: { x: "1.5em" },
          title: "Key",
          value: attn.k,
        }),
        h(SimpleWeight, {
          groupProps: { transform: "translate(100, 25)" },
          titleProps: { x: "1em" },
          title: "Query",
          value: attn.q,
        }),
        h(SimpleWeight, {
          groupProps: { transform: "translate(190, 25)" },
          titleProps: { x: "1em" },
          title: "Value",
          value: attn.v,
        }),

        h(Line, { d: "M150,70, 150,80" }),
        h(LineEnd, { d: "M50,70 50,80, 150,80 150,100" }),
        h(Line, { d: "M230,70, 230,150 150,150" }),
        h(Line, { d: "M150,80, 150,100 150,100 " }),

        h(
          "g",
          { transform: "translate(120, 130)" },
          h("text", null, "Softmax"),
        ),

        h(LineEnd, { d: "M150,140, 150,160" }),

        h(SimpleWeight, {
          groupProps: { transform: "translate(115, 200)" },
          titleProps: { x: "1em" },
          title: "out",
          value: attn.out,
        }),

        h(WeightIn, {
          groupProps: { transform: "translate(115, 245)" },
          titleProps: { x: "1em" },
          title: "fc1",
          value: mlp.fc1,
        }),

        h(WeightIn, {
          groupProps: { transform: "translate(115, 330)" },
          titleProps: { x: "1em" },
          title: "fc2",
          value: mlp.fc2,
        }),
      ),
    );
  });
}

function UNetArchitecture({ layers }) {
  return [
    h(
      "div",
      { className: "unet-down" },
      Object.entries(layers.down).map(([id, layer]) => {
        let conv;
        if (layer.conv1) {
          conv = h("div", { className: "resnet-block" }, [
            h("h3", null, "ResNet Convolution"),
            h(ResNet, layer),
          ]);
        }

        let crossAttention;
        if (Object.keys(layer.attn1).length > 0) {
          crossAttention = h("div", { className: "attention-block" }, [
            h("h3", null, "Cross Attention"),
            h(CrossAttention, layer),
          ]);
        }

        let sampler;
        if (layer.conv) {
          sampler = h("div", { className: "sampler-block" }, [
            h("h3", null, "Down Sampler"),
            h(Sampler, layer),
          ]);
        }

        return h("div", null, h("h3", null, `down ${id}`), [
          crossAttention,
          conv,
          sampler,
        ]);
      }),
    ),
    h(
      "div",
      { className: "unet-mid" },
      Object.entries(layers.mid).map(([id, layer]) => {
        let conv;
        if (layer.conv1) {
          conv = h("div", { className: "resnet-block" }, [
            h("h3", null, "ResNet Convolution"),
            h(ResNet, layer),
          ]);
        }

        let crossAttention;
        if (Object.keys(layer.attn1).length > 0) {
          crossAttention = h("div", { className: "attention-block" }, [
            h("h3", null, "Cross Attention"),
            h(CrossAttention, layer),
          ]);
        }

        let sampler;
        if (layer.conv) {
          sampler = h("div", { className: "sampler-block" }, [
            h("h3", null, "Sampler"),
            h(Sampler, layer),
          ]);
        }
        return h("div", null, h("h3", null, `mid ${id}`), [
          crossAttention,
          conv,
          sampler,
        ]);
      }),
    ),
    h(
      "div",
      { className: "unet-up" },
      Object.entries(layers.up).map(([id, layer]) => {
        let conv;
        if (layer.conv1) {
          conv = h("div", { className: "resnet-block" }, [
            h("h3", null, "ResNet Convolution"),
            h(ResNet, layer),
          ]);
        }

        let crossAttention;
        if (Object.keys(layer.attn1).length > 0) {
          crossAttention = h("div", { className: "attention-block" }, [
            h("h3", null, "Cross Attention"),
            h(CrossAttention, layer),
          ]);
        }

        let sampler;
        if (layer.conv) {
          sampler = h("div", { className: "sampler-block" }, [
            h("h3", null, "Up Sampler"),
            h(Sampler, layer),
          ]);
        }
        return h("div", null, h("h3", null, `up ${id}`), [
          crossAttention,
          conv,
          sampler,
        ]);
      }),
    ),
  ];
}

function MultiLayerPerception({ fc1, fc2 }) {
  // console.log("MLP", fc1, fc2);
  return h("div", null, h("div", null, fc1), h("div", null, fc2));
}

function Attention({ k, q, v, out }) {
  // console.log("Attention", k, q, v, out);
  // k + q => softmax (*v) -> proj
  return h(
    "div",
    null,
    h("div", null, [
      h(MetaAttribute, { name: "Key", value: k }),
      h(MetaAttribute, { name: "Query", value: q }),
    ]),
    h("div", null, h(MetaAttribute, { name: "Value", value: v })),
    h("div", null, h(MetaAttribute, { name: "Out Proj", value: out })),
  );
}

function CrossAttention({ proj_in, attn1, attn2, ff1, ff2, proj_out }) {
  return h(
    "svg",
    { className: "cross-attention-layer", width: "18em", height: "915" },
    h(
      "defs",
      null,
      h(
        "marker",
        {
          id: "head",
          orient: "auto",
          markerWidth: 3,
          markerHeight: 4,
          refX: "0.1",
          refY: "2",
        },
        h("path", { d: "M0,0 V4 L2,2 Z", fill: "currentColor" }),
      ),
      // h(
      //   "filter",
      //   { id: "filter1", x: 0, y: 0 },
      //   h("feOffset", {
      //     result: "offOut",
      //     in: "SourceAlpha",
      //     dx: 1,
      //     dy: 1,
      //   }),
      //   h("feGaussianBlur", {
      //     result: "blurOut",
      //     in: "offOut",
      //     stdDeviation: 1,
      //   }),
      //   h("feBlend", {
      //     in: "SourceGraphic",
      //     in2: "blurOut",
      //     mode: "normal",
      //   }),
      // ),
    ),
    h(SimpleWeight, {
      groupProps: {
        className: "cross-attention-proj-in",
        transform: "translate(125, 25)",
      },
      titleProps: {
        x: "0em",
      },
      title: "Proj in",
      value: proj_in,
    }),
    h(
      Group,
      {
        className: "attention-flow proj-to-k-q-v",
        transform: "translate(60, 70)",
      },
      h(Line, { d: "M100,0, 100,10" }),
      h(Line, { d: "M0,10, 195,10" }),
      h(LineEnd, { d: "M2,10, 2,30" }),
      h(LineEnd, { d: "M100,10, 100,30" }),
      h(LineEnd, { d: "M193,10, 193,30" }),
    ),
    // self attention k q v
    h(
      Group,
      { className: "", transform: "translate(25, 130)" },
      h(SimpleWeight, {
        titleProps: { x: "1em" },
        title: "Key",
        value: attn1.k,
      }),
      h(SimpleWeight, {
        groupProps: { transform: "translate(100, 0)" },
        title: "Query",
        value: attn1.q,
      }),
      h(Line, { d: "M45,45 45,60 135,60 135,45" }),
      h(LineEnd, { d: "M85,60 85,70" }),
    ),
    h(SimpleWeight, {
      groupProps: {
        className: "attention-value",
        transform: "translate(220, 130)",
      },
      title: "Value",
      value: attn1.v,
    }),
    // h(GText, { x: "0.5em", title: "Softmax" }, "Softmax"),
    // h(GText, { x: "0.5em", title: "Add + Norm" }, "Add + Norm"),
    h(LineEnd, { d: "M250,165 250,250 190,250" }),
    h(LineEnd, { d: "M110,265 110,275" }),
    // Self attention to cross attention
    // h(
    //   Group,
    //   { transform: "translate(25, 130)" },
    //   h(Weight, { title: "Key", value: "0.02758" }),
    //   h(Weight, { title: "Query", value: "0.02758" }),
    //   h(Line, {
    //     className: "key-query-softmax",
    //     d: "M45,45 45,60 135,60 135,45",
    //   }),
    //   h(LineEnd, {
    //     d: "M85,60 85,70",
    //   }),
    // ),
    // h(Weight, {
    //   groupProps: { transform: "translate(220 130)" },
    //   title: "Value",
    //   value: "0.02758",
    // }),
    h(
      GText,
      {
        groupProps: { transform: "translate(70 225)" },
        x: "0.5em",
      },
      "Softmax",
    ),
    h(
      GText,
      {
        groupProps: { transform: "translate(60 255)" },
        x: "0.5em",
      },
      "Add + Norm",
    ),
    h(LineEnd, {
      className: "value-add-softmax",
      d: "M250,165, 250,250 190,250",
    }),
    h(LineEnd, { className: "value-add-softmax", d: "M110,265 110,275" }),

    h(LineEnd, {
      className: "out-to-norm",
      d: "M150,315, 290,315 290,565 190,565",
    }),
    h(LineEnd, { className: "out-to-query", d: "M160,315, 160,395" }),

    // Cross attention from text encoder

    h(Group, { transform: "translate(0, 368)" }, h("text", null, "From TE")),
    h(
      Group,
      { className: "cross-attention-group", transform: "translate(0, 375)" },
      h(LineEnd, { d: "M0,0 255,0 255,20" }),
      h(LineEnd, { d: "M60,0 60,20" }),
    ),

    h(SimpleWeight, {
      groupProps: {
        className: "attention-out",
        transform: "translate(75, 310)",
      },
      titleProps: {
        x: "1em",
      },
      title: "Out",
      value: attn1.out,
    }),

    // CROSS ATTENTION
    h(
      Group,
      null,
      h(SimpleWeight, {
        groupProps: {
          className: "attention-key",
          transform: "translate(25, 425)",
        },
        titleProps: {
          x: "1em",
        },
        title: "Key",
        value: attn2.k,
      }),

      h(SimpleWeight, {
        groupProps: {
          className: "attention-query",
          transform: "translate(125, 425)",
        },
        title: "Query",
        value: attn2.q,
      }),

      h(Line, { d: "M60,470 60,480 160,480 160,470" }),
      h(LineEnd, { d: "M110,480 110,490" }),
    ),
    h(
      Group,
      {
        transform: "translate(220, 425)",
      },
      h(SimpleWeight, {
        groupProps: {
          className: "attention-value",
        },
        title: "Value",
        value: attn2.v,
      }),
      h(LineEnd, { d: "M30,40 30,115 -30,115" }),
    ),

    h(
      Group,
      { transform: "translate(70, 515)" },
      h("text", { title: "Softmax", x: "0.5em" }, "Softmax"),
      h("text", { title: "Add + Norm", y: "1.5em" }, "Add + Norm"),
      h("text", { title: "Add + Norm", y: "3em" }, "Add + Norm"),
    ),

    h(WeightIn, {
      groupProps: {
        transform: "translate(75, 590)",
      },
      titleProps: {
        x: "1em",
      },
      title: "Out",
      value: attn2.out,
    }),
    h(WeightIn, {
      groupProps: {
        transform: "translate(75, 675)",
      },
      titleProps: {
        x: "-1em",
      },
      title: "ff_net_0_proj",
      value: ff1,
    }),
    h(WeightIn, {
      groupProps: {
        transform: "translate(75, 760)",
      },
      title: "ff_net_2",
      value: ff2,
    }),

    h(WeightIn, {
      groupProps: {
        className: "cross-attention-proj-out",
        transform: "translate(75, 840)",
      },
      title: "Proj out",
      value: proj_out,
    }),
    // h(
    //   "g",
    //   { className: "cross-attention-proj-in", transform: "translate(125,25)" },
    //   h("text", { title: "Proj in" }, "Proj in"),
    //   h("text", { title: "Proj in", y: "1.5em" }, "0.02758"),
    // ),
  );
}

function ResNet({ conv1, time_emb_proj, conv2, conv_shortcut }) {
  return h(
    "svg",
    { className: "resnet-conv-layer", width: "7em", height: "340" },
    h(
      "defs",
      null,
      h(
        "marker",
        {
          id: "head",
          orient: "auto",
          markerWidth: 3,
          markerHeight: 4,
          refX: "0.1",
          refY: "2",
        },
        h("path", { d: "M0,0 V4 L2,2 Z", fill: "currentColor" }),
      ),
    ),
    [
      h(WeightIn, {
        groupProps: {
          transform: "translate(30, 0)",
        },
        titleProps: {
          x: "0.5em",
        },
        title: "conv1",
        value: conv1,
      }),
      h(WeightIn, {
        groupProps: {
          transform: "translate(30, 85)",
        },
        titleProps: {
          x: "-1.4em",
        },
        title: "time_emb_proj",
        value: time_emb_proj,
      }),
      h(WeightIn, {
        groupProps: {
          transform: "translate(30, 170)",
        },
        titleProps: {
          x: "0.5em",
        },
        title: "conv2",
        value: conv2,
      }),
      h(WeightIn, {
        groupProps: {
          transform: "translate(30, 255)",
        },
        titleProps: {
          x: "-1.3em",
        },
        title: "conv_shortcut",
        value: conv_shortcut,
      }),
    ],
  );
}

function Sampler({ conv }) {
  return h(
    "svg",
    { className: "sampler-layer", width: "4em", height: "85" },
    h(
      "defs",
      null,
      h(
        "marker",
        {
          id: "head",
          orient: "auto",
          markerWidth: 3,
          markerHeight: 4,
          refX: "0.1",
          refY: "2",
        },
        h("path", { d: "M0,0 V4 L2,2 Z", fill: "currentColor" }),
      ),
    ),
    [
      h(WeightIn, {
        groupProps: {
          transform: "translate(0, 0)",
        },
        titleProps: {
          x: "0.75em",
        },
        title: "conv",
        value: conv,
      }),
    ],
  );
}

function Group(props) {
  return h("g", props);
}

function Line({ d, ...rest }) {
  return h("path", {
    markerEnd: "none",
    strokeWidth: 4,
    fill: "none",
    stroke: "currentColor",
    // filter: "url(#filter1)",
    d,
    ...rest,
  });
}

function LineEnd(props) {
  return h(Line, {
    markerEnd: "url(#head)",
    // filter: "url(#filter1)",
    ...props,
  });
}

function GText({ children, groupProps, ...rest }) {
  return h("g", groupProps, h("text", rest, children));
}

function WeightIn({ groupProps, titleProps, valueProps, title, value }) {
  return h(
    "g",
    groupProps,
    h("path", {
      markerEnd: "url(#head)",
      strokeWidth: 4,
      fill: "none",
      stroke: "currentColor",
      d: "M36,0 36,10",
    }),
    h("text", { ...(titleProps ?? []), title: title, y: "2em" }, title),
    h(
      "text",
      {
        x: "0.5em",
        y: "3.5em",
        ...(valueProps ?? []),
        title: title,
      },
      value?.toPrecision(4),
    ),
  );
}

function SimpleWeight({ groupProps, titleProps, valueProps, title, value }) {
  return h(
    "g",
    { className: title, ...groupProps },
    h(
      "text",
      {
        x: "0.5em",
        ...(titleProps ?? []),
        title: title,
      },
      title,
    ),
    h(
      "text",
      {
        x: "0.5em",
        y: "1.5em",
        ...(valueProps ?? []),
        title: title,
      },
      value?.toPrecision(4),
    ),
  );
}

function Main({ metadata, filename }) {
  if (!metadata) {
    return h(
      "main",
      null,
      h("div", null, "No metadata for this file"),
      h(Headline, { filename }),
      h(
        "div",
        { className: "row space-apart" },
        h(LoRANetwork, { metadata }),
        h(Precision),
      ),
      h(Weight, { metadata, filename }),
      h(Advanced, { metadata, filename }),
    );
  }

  return h(
    "main",
    null,
    h(PretrainedModel, { metadata }),
    h(Network, { metadata }),
    h(LRScheduler, { metadata }),
    h(Optimizer, { metadata }),
    h(Weight, { metadata, filename }),
    h(EpochStep, { metadata }),
    h(Batch, { metadata }),
    h(Noise, { metadata }),
    h(Loss, { metadata }),
    h(CaptionDropout, { metadata }),
    h(Dataset, { metadata }),
    h(Advanced, { metadata, filename }),
  );
}

function Raw({ metadata, filename }) {
  const [showRaw, setShowRaw] = React.useState(undefined);

  if (showRaw) {
    const entries = Object.fromEntries(metadata);

    const sortedEntries = Object.keys(entries)
      .sort()
      .reduce((obj, key) => {
        obj[key] = entries[key];
        return obj;
      }, {});

    return h("div", { className: "full-overlay" }, [
      h("pre", { key: "pre" }, JSON.stringify(sortedEntries, null, 2)),

      h("div", { className: "action-overlay", key: "action overlay" }, [
        h(
          "button",
          {
            className: "download",
            key: "download button",
            onClick: () => {
              sortedEntries;
              const data =
                "text/json;charset=utf-8," +
                encodeURIComponent(JSON.stringify(sortedEntries, null, 2));
              const a = document.createElement("a");
              a.href = "data:" + data;
              a.download = `${filename.replace(
                ".safetensors",
                "",
              )}-metadata.json`;

              var container = document.body;
              container.appendChild(a);
              a.click();

              a.remove();
            },
          },
          "Download",
        ),
        h(
          "button",
          {
            className: "close",
            key: "close button",
            onClick: () => {
              setShowRaw(false);
            },
          },
          "Close",
        ),
      ]),
    ]);
  }

  return h(
    "div",
    {
      style: {
        display: "grid",
        justifyItems: "end",
        alignItems: "flex-start",
      },
    },
    h(
      "button",
      {
        className: "secondary",
        onClick: () => {
          setShowRaw(true);
        },
      },
      "Raw metadata",
    ),
  );
}

function Headline({ metadata, filename }) {
  let raw;
  if (metadata) {
    raw = h(Raw, { key: "raw", metadata, filename });
  }

  return h("div", { className: "headline" }, [
    h("div", { key: "lora file" }, "LoRA file"),

    raw,
    h("h1", { key: "filename" }, filename),
  ]);
}

function Support() {
  const [modal, setModal] = React.useState(false);

  React.useEffect(() => {
    if (!modal) {
      return;
    }

    function close(e) {
      // escape
      if (e.keyCode == 27) {
        setModal(false);

        window.removeEventListener("keydown", close);
      }
    }
    window.addEventListener("keydown", close);

    return function cleanup() {
      window.removeEventListener("keydown", close);
    };
  }, [modal]);

  if (modal) {
    return h(
      "div",
      { className: "modal" },
      h(
        "div",
        {},
        h(
          "div",
          { style: { textAlign: "right", padding: "1em" } },
          h(
            "button",
            {
              onClick: () => {
                setModal(false);
              },
            },
            "Close",
          ),
        ),
        h(
          "p",
          { className: "primary-text" },
          "Primary support through ",
          h(
            "a",
            {
              href: "https://github.com/rockerBOO/lora-inspector-rs/issues",
              target: "_blank",
            },

            "Github Issues",
          ),
        ),
        h(
          "p",
          { className: "primary-text" },
          "Looking to give support for this project? Through ",
          h(
            "a",
            {
              href: "https://github.com/sponsors/rockerBOO",
              target: "_blank",
            },
            "Github Sponsors",
          ),
        ),
        h(
          "p",
          { className: "primary-text" },
          "Come see the source code over on ",
          h(
            "a",
            {
              href: "https://github.com/rockerBOO/lora-inspector-rs",
              target: "_blank",
            },
            "Github (lora-inspector-rs)",
          ),
        ),
        h(
          "p",
          {
            className: "primary-text",
          },
          "Thank you for your support! - Dave (rockerBOO)",
        ),
      ),
    );
  }

  return h(
    "button",
    {
      onClick: () => {
        setModal(modal ? false : true);
      },
    },
    `Support`,
  );
}

function NoMetadata({ filename }) {
  return h(
    "main",
    null,

    h(Headline, { filename }),
    [h(Weight, { filename })],
  );
}

function Metadata({ metadata, filename }) {
  if (!metadata) {
    return h(Main, { metadata, filename });
  }

  return [
    h(Headline, { key: "headline", metadata, filename }),
    h(Header, { key: "header", metadata, filename }),
    h(Main, { key: "main", metadata, filename }),
  ];
}

const isAdvancedUpload = (function () {
  var div = document.createElement("div");
  return (
    ("draggable" in div || ("ondragstart" in div && "ondrop" in div)) &&
    "FormData" in window &&
    "FileReader" in window
  );
})();

if (isAdvancedUpload) {
  document.querySelector("#dropbox").classList.add("has-advanced-upload");
}

const dropbox = document.querySelector("#dropbox");

[
  "drag",
  "dragstart",
  "dragend",
  "dragover",
  "dragenter",
  "dragleave",
  // "drop",
].forEach((evtName) =>
  dropbox.addEventListener(evtName, (e) => {
    e.preventDefault();
    e.stopPropagation();
  }),
);

["dragover", "dragenter"].forEach((evtName) => {
  dropbox.addEventListener(evtName, () => {
    dropbox.classList.add("is-dragover");
  });
});

["dragleave", "dragend", "drop"].forEach((evtName) => {
  dropbox.addEventListener(evtName, () => {
    dropbox.classList.remove("is-dragover");
  });
});

// ["drop"].forEach((evtName) => {
//   document.addEventListener(evtName, async (e) => {
//     e.preventDefault();
//   });
// });

const files = new Map();
let mainFilename;

const workers = new Map();

async function addWorker(file) {
  const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
  // const worker = new InspectorWorker();

  workers.set(file, worker);

  return new Promise((resolve, reject) => {
    const timeouts = [];
    const worker = workers.get(file);

    worker.onmessage = (event) => {
      timeouts.map((timeout) => clearTimeout(timeout));
      worker.onmessage = undefined;
      resolve(worker);
    };

    function checkIfAvailable() {
      worker.postMessage({ messageType: "is_available", reply: true });
      const timeout = setTimeout(() => {
        checkIfAvailable();
      }, 200);

      timeouts.push(timeout);
    }

    checkIfAvailable();
  });
}

function getWorker(file) {
  return workers.get(file);
}

function removeWorker(file) {
  const worker = workers.get(file);
  worker.terminate();
  return workers.delete(file);
}

function clearWorkers() {
  Array.from(workers.keys()).forEach((key) => {
    removeWorker(key);
  });
}

init().then(() => {
  ["drop"].forEach((evtName) => {
    document.addEventListener(evtName, async (e) => {
      e.preventDefault();
      e.stopPropagation();

    const droppedFiles = e.dataTransfer.files;
    for (let i = 0; i < droppedFiles.length; i++) {
      if (files.item(i).type != "") {
        addErrorMessage("Invalid filetype. Try a .safetensors file.");
        continue;
      }

      processFile(droppedFiles.item(i));
    }
  });
});

document.querySelector("#file").addEventListener("change", async function (e) {
  e.preventDefault();
  e.stopPropagation();

  const files = e.target.files;

  for (let i = 0; i < files.length; i++) {
    if (files.item(i).type != "") {
      addErrorMessage("Invalid filetype. Try a .safetensors file.");
      continue;
    }

    processFile(files.item(i));
  }
});
// });

async function handleMetadata(metadata, filename) {
  dropbox.classList.remove("box__open");
  dropbox.classList.add("box__closed");
  document.querySelector(".support").classList.remove("hidden");
  document.querySelector(".home")?.classList.remove("home");
  document.querySelector(".box").classList.remove("box__open");
  document.querySelector(".box__intro").classList.add("hidden");
  document.querySelector(".note").classList.add("hidden");
  const root = ReactDOM.createRoot(document.getElementById("results"));
  root.render(
    h(Metadata, {
      metadata,
      filename,
    }),
  );
}

(() => {
  const root = ReactDOM.createRoot(document.querySelector(".support"));
  root.render(h(Support, {}));
})();

let uploadTimeoutHandler;

async function processFile(file) {
  clearWorkers();
  console.log("adding worker", file.name);
  const worker = await addWorker(file.name);

  // console.log("terminate", file.name)
  //  terminatePreviousProcessing(file.name);

  mainFilename = undefined;

  console.log("Uploading file", file.name);

  worker.postMessage({ messageType: "file_upload", file: file });
  processingMetadata = true;
  const cancel = loading(file.name);

  uploadTimeoutHandler = setTimeout(() => {
    cancel();
    addErrorMessage("Timeout loading LoRA. Try again.");
  }, 12000);

  function messageHandler(e) {
    // console.log('got message on main', e.data)
    clearTimeout(uploadTimeoutHandler);
    if (e.data.messageType === "metadata") {
      processingMetadata = false;

      // Setup some access points to the file
      // (we shouldn't hold on to the file handlers but just he metadata)
      files.set(file.name, file);
      mainFilename = e.data.filename;

      handleMetadata(e.data.metadata, file.name).then(() => {
        worker.postMessage({
          messageType: "network_module",
          name: mainFilename,
        });
        worker.postMessage({ messageType: "network_args", name: mainFilename });
        worker.postMessage({ messageType: "network_type", name: mainFilename });
        worker.postMessage({ messageType: "weight_keys", name: mainFilename });
        worker.postMessage({ messageType: "alpha_keys", name: mainFilename });
        worker.postMessage({ messageType: "base_names", name: mainFilename });
        worker.postMessage({ messageType: "weight_norms", name: mainFilename });
      });
      finishLoading();
    } else {
      // console.log("UNHANDLED MESSAGE", e.data);
    }
  }

  worker.addEventListener("message", messageHandler);
}

// if we are processing the uploaded file
// we want to be able to terminate the worker if we are still working on a previous file
// in the current implementation
let processingMetadata = false;
function terminatePreviousProcessing(file) {
  const worker = getWorker(file);
  if (processingMetadata) {
    // restart the worker
    worker.terminate();
    // make a new worker
    removeWorker(worker);
    addWorker(file);
  }

  processingMetadata = false;
}

function cancelLoading(file) {
  terminatePreviousProcessing(file);
  finishLoading();
  clearTimeout(uploadTimeoutHandler);
}

window.addEventListener("keyup", (e) => {
  if (e.key === "Escape") {
    cancelLoading(file);
  }
});

function loading(file) {
  const loadingEle = document.createElement("div");
  const loadingOverlayEle = document.createElement("div");

  loadingEle.classList.add("loading-file");
  loadingOverlayEle.classList.add("loading-overlay");
  loadingOverlayEle.id = "loading-overlay";

  loadingEle.textContent = "loading...";
  loadingOverlayEle.appendChild(loadingEle);
  document.body.appendChild(loadingOverlayEle);

  let clicks = 0;
  loadingOverlayEle.addEventListener("click", () => {
    clicks += 1;

    // The user is getting fustrated or we are about to make them made. Either way.
    if (clicks > 1) {
      cancelLoading(file);
    }
  });

  return function cancel() {
    cancelLoading(file);
  };
}

function finishLoading() {
  const overlay = document.querySelector("#loading-overlay");

  if (overlay) {
    overlay.remove();
  }
}

function addErrorMessage(errorMessage) {
  const errorEle = document.createElement("div");
  const errorOverlayEle = document.createElement("div");
  const errorBlockEle = document.createElement("div");

  errorEle.classList.add("error");
  errorBlockEle.classList.add("error-block");
  errorOverlayEle.classList.add("error-overlay");
  errorOverlayEle.id = "error-overlay";

  errorEle.textContent = errorMessage;

  const button = document.createElement("button");
  button.textContent = "Close";
  button.addEventListener("click", closeErrorMessage);

  errorBlockEle.append(errorEle, button);

  errorBlockEle.addEventListener("click", (e) => {
    e.preventDefault();
    e.stopPropagation();
  });

  errorOverlayEle.appendChild(errorBlockEle);

  errorOverlayEle.addEventListener("click", (e) => {
    e.preventDefault();
    closeErrorMessage();
  });

  document.body.appendChild(errorOverlayEle);
}

function closeErrorMessage() {
  const overlay = document.querySelector("#error-overlay");

  if (overlay) {
    overlay.remove();
  }
}

async function trySyncMessage(message, file, matches = []) {
  const worker = getWorker(file);
  return new Promise((resolve) => {
    worker.postMessage({ ...message, reply: true });

    const workerHandler = (e) => {
      if (matches.length > 0) {
        const hasMatches =
          matches.filter((match) => e.data[match] === message[match]).length ===
          matches.length;
        if (hasMatches) {
          worker.removeEventListener("message", workerHandler);
          resolve(e.data);
        }
      } else if (e.data.messageType === message.messageType) {
        worker.removeEventListener("message", workerHandler);
        resolve(e.data);
      }
    };

    worker.addEventListener("message", workerHandler);
  });
}

async function listenProgress(messageType, file) {
  const worker = getWorker(file);
  let isFinished = false;
  function finishedWorkerHandler(e) {
    if (e.data.messageType === `${messageType}_finished`) {
      worker.removeEventListener("message", finishedWorkerHandler);
      isFinished = true;
    }
  }

  worker.addEventListener("message", finishedWorkerHandler);

  return async function* listen() {
    if (isFinished) {
      return;
    }

    yield await new Promise((resolve) => {
      function workerHandler(e) {
        if (e.data.messageType === messageType) {
          worker.removeEventListener("message", workerHandler);
          resolve(e.data);
        }
      }

      worker.addEventListener("message", workerHandler);
    });
  };
}
