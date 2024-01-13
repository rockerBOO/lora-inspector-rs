const h = React.createElement;

function Header({ metadata }) {
  return h("header", null, h(ModelSpec, { metadata }));
}

function ModelSpec({ metadata }) {
  if (!metadata.has("modelspec.title")) {
    return null;
  }

  return h(
    "div",
    { className: "model-spec" },
    h("div", { className: "row space-apart" }, [
      h(MetaAttribute, {
        name: "Date",
        value: new Date(metadata.get("modelspec.date")).toLocaleString(),
      }),
      h(MetaAttribute, {
        name: "Title",
        value: metadata.get("modelspec.title"),
      }),
      h(MetaAttribute, {
        name: "Prediction type",
        value: metadata.get("modelspec.prediction_type"),
      }),
    ]),
    h("div", { className: "row space-apart" }, [
      h(MetaAttribute, {
        name: "License",
        value: metadata.get("modelspec.license"),
      }),
      h(MetaAttribute, {
        name: "Description",
        value: metadata.get("modelspec.description"),
      }),

      h(MetaAttribute, { name: "Tags", value: metadata.get("modelspec.tags") }),
    ]),

    metadata.has("ss_training_comment") &&
      h(
        "div",
        { className: "row space-apart" },
        h(MetaAttribute, {
          name: "Training comment",
          value: metadata.get("ss_training_comment"),
        }),
      ),
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
    h("div", {}, [
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
    ]),
    h("div", {}, [
      h(MetaAttribute, {
        name: "Session ID",
        value: metadata.get("ss_session_id"),
      }),
      h(MetaAttribute, {
        name: "sd-scripts commit hash",
        value: metadata.get("ss_sd_scripts_commit_hash"),
        valueClassName: "hash",
      }),
    ]),
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

function MetaAttribute({ name, value, valueClassName, metadata }) {
  return h(
    "div",
    {},
    h("div", { title: name, className: "caption" }, name),
    h("div", { title: name, className: valueClassName ?? "" }, value),
  );
}

function Network({ metadata, filename }) {
  const [networkModule, setNetworkModule] = React.useState(
    metadata.get("ss_network_module"),
  );
  const [networkType, setNetworkType] = React.useState("");
  const [networkArgs, setNetworkArgs] = React.useState(
    metadata.has("ss_network_args")
      ? JSON.parse(metadata.get("ss_network_args"))
      : null,
  );

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
  }, []);

  let networkOptions;

  if (networkType === "DiagOFT") {
    networkOptions = h(DiagOFTNetwork, { metadata });
  } else {
    networkOptions = h(LoRANetwork, { metadata });
  }

  return [
    h(
      "div",
      { className: "row space-apart" }, //
      h(MetaAttribute, {
        name: "Network module",
        value: networkModule,
      }),
      h(MetaAttribute, {
        name: "Network type",
        value: networkType,
      }),
      networkOptions,
      h(MetaAttribute, {
        name: "Network dropout",
        valueClassName: "number",
        value: metadata.get("ss_network_dropout"),
      }),
    ),
    h(
      "div",
      { className: "row space-apart" },
      h(MetaAttribute, {
        name: "Network args",
        valueClassName: "args",
        value: JSON.stringify(networkArgs),
      }),
    ),
    // h("div", {}, [
    //   h("div", { title: "seed" }, metadata.get("ss_seed")),
    //   h("div", { title: "Training started at" }, trainingStart),
    //   h("div", { title: "Training ended at" }, trainingEnded),
    // ]),
  ];
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

function LoRANetwork({ metadata }) {
  const [alphas, setAlphas] = React.useState([
    (metadata && metadata.get("ss_network_alpha")) ?? undefined,
  ]);
  const [dims, setDims] = React.useState([
    (metadata && metadata.get("ss_network_dim")) ?? undefined,
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
      valueClassName: "rank",
      value: dims.join(", "),
    }),
    h(MetaAttribute, {
      name: "Network Alpha",
      valueClassName: "alpha",
      value: alphas
        .map((alpha) => {
          if (typeof alpha === "number") {
            return alpha.toPrecision(2);
          } else if (alpha.includes(".")) {
            return parseFloat(alpha).toPrecision(2);
          } else {
            return parseInt(alpha);
          }
        })
        .join(", "),
    }),
  ];
}

function LRScheduler({ metadata }) {
  const lrScheduler = metadata.has("ss_lr_scheduler_type")
    ? metadata.get("ss_lr_scheduler_type")
    : metadata.get("ss_lr_scheduler");

  return [
    h(
      "div",
      { className: "row space-apart" },
      h(MetaAttribute, { name: "LR Scheduler", value: lrScheduler }),
    ),
    h("div", { className: "row space-apart" }, [
      h(MetaAttribute, {
        name: "Learning Rate",
        valueClassName: "lr number",
        value: metadata.get("ss_learning_rate"),
      }),
      metadata.has("ss_lr_warmup_steps") &&
        h(MetaAttribute, {
          name: "Learning Rate",
          valueClassName: "lr number",
          value: metadata.get("ss_lr_warmup_steps"),
        }),
      h(MetaAttribute, {
        name: "UNet Learning Rate",
        valueClassName: "lr number",
        value: metadata.get("ss_unet_lr"),
      }),
      h(MetaAttribute, {
        name: "Text Encoder Learning Rate",
        valueClassName: "lr number",
        value: metadata.get("ss_text_encoder_lr"),
      }),
    ]),
  ];
}

function Optimizer({ metadata }) {
  return h("div", { className: "row space-apart" }, [
    h(MetaAttribute, {
      name: "Optimizer",
      value: metadata.get("ss_optimizer"),
    }),
    h(MetaAttribute, {
      name: "Seed",
      valueClassName: "number",
      value: metadata.get("ss_seed"),
    }),
  ]);
}

function Weight({ metadata, filename }) {
  const [precision, setPrecision] = React.useState("");
  // const [averageStrength, setAverageStrength] = React.useState(undefined);
  // const [averageMagnitude, setAverageMagnitude] = React.useState(undefined);

  // React.useEffect(() => {
  //   setAverageStrength(get_average_strength(buffer));
  //   setAverageMagnitude(get_average_magnitude(buffer));
  // }, []);

  React.useEffect(() => {
    trySyncMessage(
      { messageType: "precision", name: mainFilename },
      mainFilename,
    ).then((resp) => {
      setPrecision(resp.precision);
    });
  }, []);

  if (!metadata) {
    return h(Blocks, { metadata, filename });
  }

  return [
    h("div", { className: "row space-apart" }, [
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
        name: "Precision",
        valueClassName: "number",
        value: precision,
      }),
      metadata.has("ss_full_fp16") &&
        h(MetaAttribute, {
          name: "Full fp16",
          valueClassName: "number",
          value: metadata.get("ss_full_fp16"),
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
    ]),
    h(Blocks, { metadata, filename }),
  ];
}

// CHART.JS DEFAULTS
// Chart.defaults.font.size = 16;
// Chart.defaults.font.family = "monospace";

function scale_weight() {
  // get base_names
  // get scale weight
  // get progress
}

function Blocks({ metadata, filename }) {
  // console.log("!!!! BLOCKS !!!!! METADATA FILENAME", filename);
  const [hasBlockWeights, setHasBlockWeights] = React.useState(false);
  const [teMagBlocks, setTEMagBlocks] = React.useState(new Map());
  const [unetMagBlocks, setUnetMagBlocks] = React.useState(new Map());
  const [normProgress, setNormProgress] = React.useState(0);
  const [currentCount, setCurrentCount] = React.useState(0);
  const [totalCount, setTotalCount] = React.useState(0);
  const [startTime, setStartTime] = React.useState(undefined);
  const [currentBaseName, setCurrentBaseName] = React.useState("");
  const [canHaveBlockWeights, setCanHaveBlockWeights] = React.useState(false);

  const [scaleWeightProgress, setScaleWeightProgress] = React.useState(0);
  const [currentScaleWeightCount, setCurrentScaleWeightCount] =
    React.useState(0);
  const [totalScaleWeightCount, setTotalScaleWeightCount] = React.useState(0);

  // setCurrentScaleWeightCount(value.currentCount);
  // setTotalScaleWeightCount(value.totalCount);
  // setScaleWeightProgress(value.currentCount / value.totalCount);

  const teChartRef = React.useRef(null);
  const unetChartRef = React.useRef(null);

  React.useEffect(() => {
    if (!hasBlockWeights) {
      return;
    }

    trySyncMessage(
      {
        messageType: "scale_weights_with_progress",
        name: filename,
        reply: true,
      },
      filename,
    ).then(() => {
      console.log("getting l2 norms...");

      listenProgress("l2_norms_progress", filename).then(
        async (getProgress) => {
          while ((progress = await getProgress().next())) {
            const value = progress.value;
            setCurrentBaseName(value.baseName);
            setCurrentCount(value.currentCount);
            setTotalCount(value.totalCount);
            setNormProgress(value.currentCount / value.totalCount);
          }
        },
      );

      trySyncMessage(
        {
          messageType: "l2_norm",
          name: filename,
          reply: true,
        },
        filename,
      ).then((resp) => {
        // console.log(resp);
        // setTEMagBlocks(averageMagnitudes.get("text_encoder"));
        setTEMagBlocks(resp.norms.te);
        setUnetMagBlocks(resp.norms.unet);
        //
        // setTEStrBlocks(averageStrength.get("text_encoder"));
        // setUnetStrBlocks(averageStrength.get("unet"));
      });
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
        // Assuming networkType of none could have block weights
        resp.networkType === undefined
      ) {
        setCanHaveBlockWeights(true);
        trySyncMessage(
          {
            messageType: "precision",
            name: filename,
            reply: true,
          },
          filename,
        ).then((resp) => {
          if (resp.precision == "bf16") {
            setCanHaveBlockWeights(false);
          }
        });
      }
    });
  }, []);

  React.useEffect(() => {
    if (!hasBlockWeights) {
      return;
    }

    setStartTime(performance.now());

    listenProgress("scale_weight_progress", filename).then(
      async (getProgress) => {
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
  }, [hasBlockWeights]);

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
      const chart = new Chartist.Line(chartRef.current, data, {
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
        plugins: [
          Chartist.plugins.ctPointLabels({
            labelOffset: {
              x: 10,
              y: -10,
            },
            textAnchor: "middle",
            labelInterpolationFnc: function (value) {
              return value.toPrecision(4);
            },
          }),
        ],
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
              easing: Chartist.Svg.Easing.easeOutQuart,
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
  } else if (!hasBlockWeights) {
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
      h("h3", {}, "Text encoder block weights"),
      h("div", { ref: teChartRef, className: "chart" }),
      h(
        "div",
        { className: "block-weights text-encoder" },
        Array.from(teMagBlocks)
          .sort(([a, _], [b, _v]) => a > b)
          .map(([k, v]) => {
            return h(
              "div",
              null,
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
      h("h3", {}, "UNet block weights"),
      h("div", { ref: unetChartRef, className: "chart" }),
      h(
        "div",
        { className: "block-weights unet" },
        Array.from(unetMagBlocks).map(([k, v]) => {
          return h(
            "div",
            null,
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

    if (currentCount === 0) {
      const elapsedTime = performance.now() - startTime;
      const perSecond = currentScaleWeightCount / (elapsedTime / 1_000);

      const remaining =
        (elapsedTime * totalScaleWeightCount) / scaleWeightProgress -
        elapsedTime * totalScaleWeightCount;
      return h(
        "div",
        { className: "block-weights-container" },
        h(
          "span",
          null,
          `Scaling weights... ${(scaleWeightProgress * 100).toFixed(
            2,
          )}% ${currentScaleWeightCount}/${totalScaleWeightCount} ${perSecond.toFixed(
            2,
          )}it/s ${(remaining / 1_000_000).toFixed(
            2,
          )}s remaining ${currentBaseName} `,
        ),
      );
    }

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
  return h("div", { className: "row space-apart part3" }, [
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
    h(MetaAttribute, {
      name: "Max Train Epochs",
      valueClassName: "number",
      value: metadata.get("ss_max_train_epochs"),
    }),
    h(MetaAttribute, {
      name: "Max Train Steps",
      valueClassName: "number",
      value: metadata.get("ss_max_train_steps"),
    }),
  ]);
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

  return h("div", { className: "row space-apart part3" }, [
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
  ]);
}

function Noise({ metadata }) {
  return h("div", { className: "row space-apart" }, [
    h(MetaAttribute, {
      name: "Noise Offset",
      valueClassName: "number",
      value: metadata.get("ss_noise_offset"),
    }),
    h(MetaAttribute, {
      name: "Adaptive Noise Scale",
      valueClassName: "number",
      value: metadata.get("ss_adaptive_noise_scale"),
    }),
    h(MultiresNoise, { metadata }),
  ]);
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
    }),
    h(MetaAttribute, {
      name: "MultiRes Noise Discount",
      valueClassName: "number",
      value: metadata.get("ss_multires_noise_discount"),
    }),
  ];
}

function Loss({ metadata }) {
  return h("div", { className: "row space-apart" }, [
    h(MetaAttribute, {
      name: "Gradient Checkpointing",
      value: metadata.get("ss_gradient_checkpointing"),
    }),
    h(MetaAttribute, {
      name: "Debiased Estimation",
      valueClassName: "number",
      value: metadata.get("ss_debiased_estimation"),
    }),
    h(MetaAttribute, {
      name: "Min SNR Gamma",
      valueClassName: "number",
      value: metadata.get("ss_min_snr_gamma"),
    }),
    h(MetaAttribute, {
      name: "Zero Terminal SNR",
      value: metadata.get("ss_zero_terminal_snr"),
    }),
    metadata.has("ss_masked_loss") &&
      h(MetaAttribute, {
        name: "Masked Loss",
        value: metadata.get("ss_masked_loss"),
      }),
  ]);
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

    datasets.map((dataset) => {
      return h(Buckets, { dataset, metadata });
    }),
  );
}

function Buckets({ dataset, metadata }) {
  return [
    h(
      "div",
      { className: "row space-apart" },
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

    h("div", null, h(BucketInfo, { metadata, dataset })),
    h(
      "h3",
      { className: "row space-apart" },

      "Subsets:",
    ),
    h(
      "div",
      { className: "subsets" },
      dataset["subsets"].map((subset) => h(Subset, { metadata, subset })),
    ),
    h("h3", {}, "Tag frequencies"),
    h(
      "div",
      { className: "tag-frequencies row space-apart" },
      Object.entries(dataset["tag_frequency"]).map(([dir, frequency]) =>
        h(
          "div",
          {},
          h("h3", {}, dir),
          h(TagFrequency, { tagFrequency: frequency, metadata }),
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
        { className: "bucket" },
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
    { className: "subset" },
    h(
      "div",
      { className: "row space-apart" },
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
    ),
    h(
      "div",
      { className: "row space-apart" },
      h(MetaAttribute, {
        name: "Flip aug",
        ...tf(subset["flip_aug"], false),
      }),
      h(MetaAttribute, {
        name: "Color aug",
        ...tf(subset["color_aug"], false),
      }),
    ),
    h(
      "div",
      { className: "row space-apart" },
      h(MetaAttribute, {
        name: "Num repeats",
        value: subset["num_repeats"],
        valueClassName: "number",
      }),
      h(MetaAttribute, {
        name: "Is reg",
        ...tf(subset["is_reg"], false),
      }),
    ),
    h(
      "div",
      { className: "row space-apart" },
      h(MetaAttribute, {
        name: "Keep tokens",
        value: subset["keep_tokens"],
        valueClassName: "number",
      }),
      h(MetaAttribute, { name: "Class token", value: subset["class_tokens"] }),
    ),
    h(
      "div",
      { className: "row space-apart" },
      h(MetaAttribute, {
        name: "shuffle caption",
        ...tf(subset["shuffle_caption"], false),
      }),
    ),
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
        { className: "tag-frequency" + alt },
        h("div", {}, count),
        h("div", {}, tag),
      );
    }),
    h(
      "div",
      null,
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
      console.log("text encoder keys", resp.textEncoderKeys);
      setTextEncoderKeys(resp.textEncoderKeys);
    });

    trySyncMessage({ messageType: "unet_keys", name: filename }, filename).then(
      (resp) => {
        resp.unetKeys.sort();
        console.log("unet keys", resp.unetKeys);
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

  return [
    h("h2", null, "Advanced"),
    h(
      "div",
      { className: "row" },
      h(
        "div",
        null,
        showBaseNames
          ? h(BaseNames, { baseNames })
          : h("div", null, `Base name keys: ${baseNames.length}`),
      ),
      h(
        "div",
        null,
        showTextEncoderKeys
          ? h(TextEncoderKeys, { textEncoderKeys })
          : h("div", null, `Text encoder keys: ${textEncoderKeys.length}`),
      ),
      h(
        "div",
        null,
        showUnetKeys
          ? h(UnetKeys, { unetKeys })
          : h("div", null, `Unet keys: ${unetKeys.length}`),
      ),
      h(
        "div",
        null,
        showAllKeys
          ? h(AllKeys, { allKeys })
          : h("div", null, `All keys: ${allKeys.length}`),
      ),
    ),
    h(Statistics, { baseNames, filename }),
  ];
}

let progress = 0;

function Statistics({ baseNames, filename }) {
  const [bases, setBases] = React.useState([]);

  React.useEffect(() => {
    if (baseNames.length === 0) {
      return;
    }

    console.time("get norms");

    trySyncMessage(
      {
        messageType: "scale_weights",
        name: filename,
        reply: true,
      },
      filename,
    ).then(() => {
      Promise.all(
        baseNames.map(async (baseName) => {
          return trySyncMessage(
            { messageType: "norms", name: filename, baseName },
            filename,
            ["messageType", "baseName"],
          ).then((resp) => {
            progress += 1;
            console.log("progress", progress / baseNames.length, resp);
            return { baseName: resp.baseName, stat: resp.norms };
          });
        }),
      ).then((bases) => {
        progress = 0;
        setBases(bases);
        console.timeEnd("get norms");
      });
    });
  }, [baseNames]);

  const teLayers = compileTextEncoderLayers(bases);
  const unetLayers = compileUnetLayers(bases);

  return [
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
          l1Norm: base.stat.get("l1_norm"),
          l2Norm: base.stat.get("l2_norm"),
          matrixNorm: base.stat.get("matrix_norm"),
          min: base.stat.get("min"),
          max: base.stat.get("max"),
          median: base.stat.get("median"),
          stdDev: base.stat.get("std_dev"),
        });
      }),
    ]),
    h(
      "div",
      { id: "te-architecture" },
      h(TEArchitecture, { layers: teLayers }),
      h(UNetArchitecture, { layers: unetLayers }),
    ),
  ];
}

function compileTextEncoderLayers(bases) {
  // we have a list of names and we want to extract the different components and put back together to use
  // with Attention

  // return [
  //   {
  //     mlp: {
  //       fc1: 0.03377176355576932,
  //     },
  //     attn: {
  //       k: 0.027610311520416056,
  //       out: 0.028784308961283214,
  //       q: 0.022726607644713133,
  //       v: 0.029905859105420967,
  //     },
  //   },
  //   {
  //     mlp: {
  //       fc1: 0.030740709804336325,
  //     },
  //     attn: {
  //       k: 0.025838870814149217,
  //       out: 0.026656885389931807,
  //       q: 0.02438057192364713,
  //       v: 0.028050367227700285,
  //     },
  //   },
  //   {
  //     mlp: {
  //       fc1: 0.028207526255732373,
  //     },
  //     attn: {
  //       k: 0.023132593414098332,
  //       out: 0.03283677896168791,
  //       q: 0.027259484438444854,
  //       v: 0.03544769447446601,
  //     },
  //   },
  //   {
  //     mlp: {
  //       fc1: 0.03271818406172145,
  //     },
  //     attn: {
  //       k: 0.02895878278258152,
  //       out: 0.027551642200681088,
  //       q: 0.0213135003023636,
  //       v: 0.03100399919716222,
  //     },
  //   },
  //   {
  //     mlp: {
  //       fc1: 0.036818285052332145,
  //     },
  //     attn: {
  //       k: 0.029139543708589184,
  //       out: 0.0327438007805441,
  //       q: 0.029182852250562816,
  //       v: 0.028927089216443012,
  //     },
  //   },
  //   {
  //     mlp: {
  //       fc1: 0.033634944570086964,
  //     },
  //     attn: {
  //       k: 0.031138590582916438,
  //       out: 0.03115476570761682,
  //       q: 0.023419985096103258,
  //       v: 0.02843812513761224,
  //     },
  //   },
  //   {
  //     mlp: {
  //       fc1: 0.03106017954275852,
  //     },
  //     attn: {
  //       k: 0.023201234995261357,
  //       out: 0.025009921446109664,
  //       q: 0.03349966167986422,
  //       v: 0.0289371964836586,
  //     },
  //   },
  //   {
  //     mlp: {
  //       fc1: 0.03536504702850587,
  //     },
  //     attn: {
  //       k: 0.02460080728650688,
  //       out: 0.030828804324378137,
  //       q: 0.029944200948548962,
  //       v: 0.026303386135356772,
  //     },
  //   },
  //   {
  //     mlp: {
  //       fc1: 0.03156469178192148,
  //     },
  //     attn: {
  //       k: 0.026440982839927758,
  //       out: 0.027864138614725913,
  //       q: 0.02647334809395087,
  //       v: 0.0323257080199173,
  //     },
  //   },
  //   {
  //     mlp: {
  //       fc1: 0.028074173055168763,
  //     },
  //     attn: {
  //       k: 0.03040033962525639,
  //       out: 0.02454701439997252,
  //       q: 0.028268590065174837,
  //       v: 0.0326967754576092,
  //     },
  //   },
  //   {
  //     mlp: {
  //       fc1: 0.028183648483636747,
  //     },
  //     attn: {
  //       k: 0.02758406973382335,
  //       out: 0.027762173771304233,
  //       q: 0.02360335896072415,
  //       v: 0.03246408533406295,
  //     },
  //   },
  //   {
  //     mlp: {
  //       fc1: 0.02499392983027854,
  //     },
  //     attn: {
  //       k: 0.022166229731455274,
  //       out: 0.03004094724091072,
  //       q: 0.027297408086285304,
  //       v: 0.026541146306402648,
  //     },
  //   },
  // ];
  const re =
    /lora_te_text_model_encoder_layers_(?<layer_id>\d+)_(?<layer_type>mlp|self_attn)_(?<sub_type>k_proj|q_proj|v_proj|out_proj|fc1|fc2)/;

  const layers = [];
  console.log(bases);

  for (const i in bases) {
    const base = bases[i];

    console.log(base);
    const match = base.baseName.match(re);

    if (match) {
      console.log(match);

      const layerId = match.groups.layer_id;
      const layerType = match.groups.layer_type;
      const subType = match.groups.sub_type;

      const layerKey = layerType === "self_attn" ? "attn" : "mlp";
      let value;

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
          subkey = "fc2";
          break;
      }

      if (!layers[layerId]) {
        layers[layerId] = {
          [layerKey]: {
            [subKey]: base.stat.get("l2_norm"),
          },
        };
      } else {
        if (!layers[layerId][layerKey]) {
          layers[layerId][layerKey] = {};
        }
        layers[layerId][layerKey][subKey] = base.stat.get("l2_norm");
      }
    }
  }

  console.log("te layers", layers);

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

      // console.log("block_type", groups["block_type"]);

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

  // return {
  //   down: {
  //     0: {
  //       proj_in: 0.7321300228392223,
  //       attn1: {
  //         k: 0.6722621767578187,
  //         q: 0.6511781156757548,
  //         v: 0.6935735581905126,
  //         out: 0.6846320232540608,
  //       },
  //       attn2: {
  //         k: 0.7510785980374958,
  //         q: 0.554719090089995,
  //         v: 0.7270581271746418,
  //         out: 0.7135258603821074,
  //       },
  //       ff1: 1.8765097058983984,
  //       ff2: 0.9652298770968581,
  //       proj_out: 0.6100351393694301,
  //     },
  //     1: {
  //       proj_in: 0.6103902379925027,
  //       attn1: {
  //         k: 0.6685348327312036,
  //         q: 0.6257657583967223,
  //         v: 0.683091543243205,
  //         out: 0.6313789984122181,
  //       },
  //       attn2: {
  //         k: 1.22194527828802,
  //         q: 0.5774264755047156,
  //         v: 0.8915744358899851,
  //         out: 0.6449179921616967,
  //       },
  //       ff1: 2.1462326275738097,
  //       ff2: 1.1291232337169568,
  //       proj_out: 0.6268967044973203,
  //     },
  //     3: {
  //       proj_in: 0.9355693016270011,
  //       attn1: {
  //         k: 0.9301549700214394,
  //         q: 1.0379488970609958,
  //         v: 1.009717654741764,
  //         out: 1.0653500916892182,
  //       },
  //       attn2: {
  //         k: 1.0423720420623361,
  //         q: 0.9328001541579402,
  //         v: 1.1016296534519077,
  //         out: 1.1686387130358198,
  //       },
  //       ff1: 2.94134593620881,
  //       ff2: 1.892295957667535,
  //       proj_out: 1.0640601767412536,
  //     },
  //     4: {
  //       proj_in: 1.0260377804008298,
  //       attn1: {
  //         k: 1.052193309234656,
  //         q: 0.9831270253796603,
  //         v: 0.9322223013871355,
  //         out: 0.9069335757919067,
  //       },
  //       attn2: {
  //         k: 0.7214671242828914,
  //         q: 0.7140062490411764,
  //         v: 0.842443850130627,
  //         out: 0.9751807725222181,
  //       },
  //       ff1: 2.7088397982534853,
  //       ff2: 1.693866276468733,
  //       proj_out: 0.8965560703858975,
  //     },
  //     6: {
  //       proj_in: 1.4359480819739419,
  //       attn1: {
  //         k: 1.7555674848632628,
  //         q: 1.6421437644890886,
  //         v: 1.6881824948501947,
  //         out: 1.530827326205961,
  //       },
  //       attn2: {
  //         k: 1.15386139350618,
  //         q: 1.5777034864715769,
  //         v: 1.3451804681553161,
  //         out: 2.0736647183530126,
  //       },
  //       ff1: 5.236047713345741,
  //       ff2: 3.3389863711246024,
  //       proj_out: 1.7424620765042624,
  //     },
  //     7: {
  //       proj_in: 1.5779324794139324,
  //       attn1: {
  //         k: 1.6173986815869084,
  //         q: 1.6172507470878852,
  //         v: 1.6295117437560875,
  //         out: 1.7009819316725845,
  //       },
  //       attn2: {
  //         k: 0.9193958945267028,
  //         q: 0.9994320157543968,
  //         v: 1.304004333276256,
  //         out: 2.1059795752494024,
  //       },
  //       ff1: 4.100873698575432,
  //       ff2: 3.0102936585498927,
  //       proj_out: 1.5443294602815303,
  //     },
  //   },
  //   mid: {
  //     0: {
  //       proj_in: 1.8603964180793588,
  //       attn1: {
  //         k: 1.65462106091712,
  //         q: 2.3652007659495387,
  //         v: 1.648676859436656,
  //         out: 1.6479986650284224,
  //       },
  //       attn2: {
  //         k: 1.4418991705038022,
  //         q: 1.4558909665630113,
  //         v: 1.1797608672857829,
  //         out: 1.8903940229242242,
  //       },
  //       ff1: 6.080725607164155,
  //       ff2: 3.7319773046131535,
  //       proj_out: 2.264789768397898,
  //     },
  //   },
  //   up: {
  //     3: {
  //       proj_in: 1.652029974755443,
  //       attn1: {
  //         k: 1.72240038562434,
  //         q: 1.7011859946657355,
  //         v: 1.7749363315623883,
  //         out: 1.9749524450062552,
  //       },
  //       attn2: {
  //         k: 1.0883090067106604,
  //         q: 1.211465910951937,
  //         v: 1.1754110696484705,
  //         out: 1.6939734699518278,
  //       },
  //       ff1: 5.058069176902104,
  //       ff2: 2.9329384409540786,
  //       proj_out: 1.5600909292652805,
  //     },
  //     4: {
  //       proj_in: 1.7350853670636872,
  //       attn1: {
  //         k: 1.7239140849805255,
  //         q: 1.8391006483593777,
  //         v: 1.599359850750919,
  //         out: 1.929919388919021,
  //       },
  //       attn2: {
  //         k: 1.2442431908742166,
  //         q: 1.332739056335861,
  //         v: 1.1531629787977997,
  //         out: 1.8005189066512577,
  //       },
  //       ff1: 5.93332855665865,
  //       ff2: 3.8249316869723895,
  //       proj_out: 1.9275357354110783,
  //     },
  //     5: {
  //       proj_in: 1.60750652461065,
  //       attn1: {
  //         k: 2.5105525270835267,
  //         q: 2.739013924387776,
  //         v: 1.6339389332378726,
  //         out: 1.7795268583263455,
  //       },
  //       attn2: {
  //         k: 1.6464602331538558,
  //         q: 1.8375230508674993,
  //         v: 1.25108014536182,
  //         out: 1.867519626511994,
  //       },
  //       ff1: 6.160500872280232,
  //       ff2: 3.337199267876231,
  //       proj_out: 1.8795627221478142,
  //     },
  //     6: {
  //       proj_in: 1.1198231032165755,
  //       attn1: {
  //         k: 1.0842911377179798,
  //         q: 0.9159201719702058,
  //         v: 1.01989556354772,
  //         out: 1.17669655283065,
  //       },
  //       attn2: {
  //         k: 0.8869824153564619,
  //         q: 0.6690626544576774,
  //         v: 0.8804039902713688,
  //         out: 1.288608631335546,
  //       },
  //       ff1: 3.1397435588562628,
  //       ff2: 1.8978276133719976,
  //       proj_out: 0.9579104441348353,
  //     },
  //     7: {
  //       proj_in: 1.1345287083438274,
  //       attn1: {
  //         k: 1.0463571336714181,
  //         q: 1.142409765711616,
  //         v: 1.0452099873430831,
  //         out: 1.0520314648763995,
  //       },
  //       attn2: {
  //         k: 0.9530294736210758,
  //         q: 0.8692950519866015,
  //         v: 0.9929179562789237,
  //         out: 1.2108682649714024,
  //       },
  //       ff1: 2.953724294213194,
  //       ff2: 1.6901417033233905,
  //       proj_out: 1.1758144352304611,
  //     },
  //     8: {
  //       proj_in: 1.1982013024938913,
  //       attn1: {
  //         k: 1.1022880685909286,
  //         q: 1.2031060730919407,
  //         v: 0.9777458175426541,
  //         out: 0.9996915531165024,
  //       },
  //       attn2: {
  //         k: 1.3256333962020932,
  //         q: 0.9910017680027595,
  //         v: 1.1635197355826374,
  //         out: 1.022877715017238,
  //       },
  //       ff1: 2.789591080215707,
  //       ff2: 1.9515492693478445,
  //       proj_out: 0.9708795562758037,
  //     },
  //     9: {
  //       proj_in: 0.5933071264684742,
  //       attn1: {
  //         k: 0.6052617076757603,
  //         q: 0.6565792771989842,
  //         v: 0.6162005816609448,
  //         out: 0.6237161937945802,
  //       },
  //       attn2: {
  //         k: 0.8634127072913751,
  //         q: 0.7537407527236594,
  //         v: 0.6707907101500438,
  //         out: 0.6343704905116913,
  //       },
  //       ff1: 1.8907164902593363,
  //       ff2: 1.0385923233363892,
  //       proj_out: 0.5587854986961417,
  //     },
  //     10: {
  //       proj_in: 0.6241509864380473,
  //       attn1: {
  //         k: 0.7197084646675959,
  //         q: 0.606873170024349,
  //         v: 0.6216814332599732,
  //         out: 0.595516753098002,
  //       },
  //       attn2: {
  //         k: 0.7361735510359417,
  //         q: 0.5444455913384406,
  //         v: 0.5662872874139346,
  //         out: 0.6262802435334639,
  //       },
  //       ff1: 1.8499362225079872,
  //       ff2: 1.011894669043252,
  //       proj_out: 0.6056375734234015,
  //     },
  //     11: {
  //       proj_in: 0.6251868405938613,
  //       attn1: {
  //         k: 0.5464932681496891,
  //         q: 0.6686424567631964,
  //         v: 1.0890761569618088,
  //         out: 0.5960820383804658,
  //       },
  //       attn2: {
  //         k: 0.9035756936601899,
  //         q: 0.7792671889783575,
  //         v: 1.1143997508429884,
  //         out: 0.6788500439665865,
  //       },
  //       ff1: 1.8967438027164611,
  //       ff2: 0.7291338396536455,
  //       proj_out: 0.6559736128196422,
  //     },
  //   },
  // };
  const re =
    /lora_unet_(down_blocks|mid_block|up_blocks)_(?<block_id>\d+)_(?<layer_type>mlp|self_attn)_(?<sub_type>k_proj|q_proj|v_proj|out_proj|fc1|fc2)/;

  // const layers = [];
  console.log(bases);

  // console.log(
  //   "bases parsed",
  //   bases.map((base) => parseSDKey(base)),
  // );

  // const layer = {
  //   proj_in: 1,
  //   attn1: { k: 1, q: 1, v: 1, out: 1 },
  //   attn2: { k: 1, q: 1, v: 1, out: 1 },
  //   ff1: 1,
  //   ff2: 1,
  //   proj_out: 1,
  // };

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
    layer = ensureLayer(layer, parsedKey.idx);

    if (parsedKey.isAttention) {
      if (base.baseName.includes("attn1")) {
        if (base.baseName.includes("to_q")) {
          layer["attn1"]["q"] = base.stat.get("l2_norm");
        } else if (base.baseName.includes("to_k")) {
          layer["attn1"]["k"] = base.stat.get("l2_norm");
        } else if (base.baseName.includes("to_v")) {
          layer["attn1"]["v"] = base.stat.get("l2_norm");
        } else if (base.baseName.includes("to_out")) {
          layer["attn1"]["out"] = base.stat.get("l2_norm");
        }
      } else if (base.baseName.includes("attn2")) {
        if (base.baseName.includes("to_q")) {
          layer["attn2"]["q"] = base.stat.get("l2_norm");
        } else if (base.baseName.includes("to_k")) {
          layer["attn2"]["k"] = base.stat.get("l2_norm");
        } else if (base.baseName.includes("to_v")) {
          layer["attn2"]["v"] = base.stat.get("l2_norm");
        } else if (base.baseName.includes("to_out")) {
          layer["attn2"]["out"] = base.stat.get("l2_norm");
        }
      } else if (base.baseName.includes("ff_net_0_proj")) {
        layer["ff1"] = base.stat.get("l2_norm");
      } else if (base.baseName.includes("ff_net_2")) {
        layer["ff2"] = base.stat.get("l2_norm");
      } else if (base.baseName.includes("proj_in")) {
        layer["proj_in"] = base.stat.get("l2_norm");
      } else if (base.baseName.includes("proj_out")) {
        layer["proj_out"] = base.stat.get("l2_norm");
      }
    } else if (parsedKey.isConv) {
      if (base.baseName.includes("conv1")) {
        layer["conv1"] = base.stat.get("l2_norm");
      } else if (base.baseName.includes("time_emb_proj")) {
        layer["time_emb_proj"] = base.stat.get("l2_norm");
      } else if (base.baseName.includes("conv2")) {
        layer["conv2"] = base.stat.get("l2_norm");
      } else if (base.baseName.includes("conv_shortcut")) {
        layer["conv_shortcut"] = base.stat.get("l2_norm");
      }
    } else if (parsedKey.isSampler) {
      if (base.baseName.includes("conv")) {
        layer["conv"] = base.state.get("l2_norm");
      }
    }
  }

  console.log("unet layers", layers);

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
    h("td", null, l1Norm.toPrecision(4)),
    h("td", null, l2Norm.toPrecision(4)),
    h("td", null, matrixNorm.toPrecision(4)),
    h("td", null, min.toPrecision(4)),
    h("td", null, max.toPrecision(4)),
    h("td", null, median.toPrecision(4)),
    h("td", null, stdDev.toPrecision(4)),
  );
}

function UnetKeys({ unetKeys }) {
  return [
    h("h3", null, "UNet keys"),
    h(
      "ul",
      null,
      unetKeys.map((unetKeys) => {
        return h("li", null, unetKeys);
      }),
    ),
  ];
}
function TextEncoderKeys({ textEncoderKeys }) {
  return [
    h("h3", null, "Text encoder keys"),
    h(
      "ul",
      null,
      textEncoderKeys.map((textEncoderKeys) => {
        return h("li", null, textEncoderKeys);
      }),
    ),
  ];
}

function BaseNames({ baseNames }) {
  return [
    h("h3", null, "Base names"),
    h(
      "ul",
      null,
      baseNames.map((baseName) => {
        return h("li", null, baseName);
      }),
    ),
  ];
}

function AllKeys({ allkeys }) {
  return [
    h("h3", null, "All keys"),
    h(
      "ul",
      null,
      allKeys.map((key) => {
        return h("li", null, key);
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
      null,
      h("h3", null, `layer ${i + 1}`),
      h(
        "svg",
        { className: "attention-layer", width: "17.5em", height: "480" },
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
        h(
          "g",
          { className: "attention-key", transform: "translate(25, 25)" },
          h("text", { title: "Key", x: "1em" }, "Key"),
          h("text", { title: "Key", y: "1.5em" }, attn.k.toPrecision(4)),
        ),
        h(
          "g",
          { className: "attention-key", transform: "translate(125, 25)" },
          h("text", { title: "Key", x: "1em" }, "Query"),
          h("text", { title: "Key", y: "1.5em" }, attn.q.toPrecision(4)),
        ),

        h(
          "g",
          { className: "attention-key", transform: "translate(220, 25)" },
          h("text", { title: "Key", x: "1em" }, "Value"),
          h("text", { title: "Key", y: "1.5em" }, attn.v.toPrecision(4)),
        ),

        h("path", {
          markerEnd: "none",
          stroke: "currentColor",
          strokeWidth: 4,
          fill: "none",
          d: "M150,80, 150,100 150,100 ",
        }),
        h("path", {
          markerEnd: "url(#head)",
          stroke: "currentColor",
          strokeWidth: 4,
          fill: "none",
          d: "M60,80 60,100, 150,100 150,120",
        }),
        h("path", {
          markerEnd: "none",
          stroke: "currentColor",
          strokeWidth: 4,
          fill: "none",
          d: "M250,80, 250,200 150,200",
        }),
        h("path", {
          markerEnd: "none",
          stroke: "currentColor",
          strokeWidth: 4,
          fill: "none",
          d: "M150,80, 150,100 150,100 ",
        }),

        h(
          "g",
          { transform: "translate(120, 150)" },
          h("text", null, "Softmax"),
        ),

        h("path", {
          markerEnd: "url(#head)",
          stroke: "currentColor",
          strokeWidth: 4,
          fill: "none",
          d: "M150,180, 150,220",
        }),

        h(
          "g",
          { className: "attention-out", transform: "translate(125 250)" },
          h("text", { title: "Out", x: "0.5em" }, "Out"),
          h(
            "text",
            { title: "Out", x: "-0.5em", y: "1.5em" },
            attn.out.toPrecision(4),
          ),
        ),

        h("path", {
          markerEnd: "url(#head)",
          stroke: "currentColor",
          strokeWidth: 4,
          fill: "none",
          d: "M150,285, 150,290",
        }),
        h(
          "g",
          { className: "mlp-fc1", transform: "translate(125, 320)" },
          h("text", { title: "Key", x: "0.5em" }, "fc1"),
          h(
            "text",
            { title: "Key", x: "-0.5em", y: "1.5em" },
            mlp.fc1.toPrecision(4),
          ),
        ),
        h("path", {
          markerEnd: "url(#head)",
          stroke: "none",
          strokeWidth: 4,
          fill: "currentColor",
          d: "M150,355, 150,360",
        }),
        h(
          "g",
          { className: "mlp-fc1", transform: "translate(125, 390)" },
          h("text", { title: "Key", x: "0.5em" }, "fc2"),
          h(
            "text",
            { title: "Key", x: "-0.5em", y: "1.5em" },
            mlp.fc2?.toPrecision(4),
          ),
        ),
      ),
      // h(Attention, attn),
      // h(MultiLayerPerception, mlp),
    );
  });
}

function UNetArchitecture({ layers }) {
  return [
    Object.entries(layers.down).map(([id, layer]) => {
      return h(
        "div",
        null,
        h("h3", null, `down ${id}`),
        layer.conv1 ? h(ResNet, layer) : h(CrossAttention, layer),
      );
    }),
    Object.entries(layers.mid).map(([id, layer]) => {
      return h(
        "div",
        null,
        h("h3", null, `mid ${id}`),
        layer.conv1 ? h(ResNet, layer) : h(CrossAttention, layer),
      );
    }),
    Object.entries(layers.up).map(([id, layer]) => {
      return h(
        "div",
        null,
        h("h3", null, `up ${id}`),
        layer.conv1 ? h(ResNet, layer) : h(CrossAttention, layer),
      );
    }),
  ];
}

function MultiLayerPerception({ fc1, fc2 }) {
  console.log("MLP", fc1, fc2);
  return h("div", null, h("div", null, fc1), h("div", null, fc2));
}

function Attention({ k, q, v, out }) {
  console.log("Attention", k, q, v, out);
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
  console.log(proj_in, attn1, attn2, ff1, ff2, proj_out);
  return h(
    "svg",
    { className: "cross-attention-layer", width: "18em", height: "930" },
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
    h(SimpleWeight, {
      groupProps: {
        className: "cross-attention-proj-in",
        transform: "translate(125, 25)",
      },
      title: "Proj in",
      value: proj_in.toPrecision(4),
    }),
    h(
      Group,
      {
        className: "attention-flow proj-to-k-q-v",
        transform: "translate(60, 70)",
      },
      h(Line, { d: "M100,0, 100,20" }),
      h(Line, { d: "M0,20, 195,20" }),
      h(LineEnd, { d: "M2,20, 2,30" }),
      h(LineEnd, { d: "M100,20, 100,30" }),
      h(LineEnd, { d: "M193,20, 193,30" }),
    ),
    // self attention k q v
    h(
      Group,
      { className: "", transform: "translate(25, 130)" },
      h(SimpleWeight, { title: "Key", value: attn1.k.toPrecision(4) }),
      h(SimpleWeight, {
        groupProps: { transform: "translate(100, 0)" },
        title: "Query",
        value: attn1.q.toPrecision(4),
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
      value: attn1.v.toPrecision(4),
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
      d: "M150,315, 310,315 310,565 190,565",
    }),
    h(LineEnd, { d: "M160,315, 160,395" }),

    // Cross attention from text encoder

    h(
      Group,
      { transform: "translate(25, 375)" },
      h(LineEnd, { d: "M0,10 225,10 225,20" }),
      h(LineEnd, { d: "M35,10 35,20" }),
    ),

    h(SimpleWeight, {
      groupProps: {
        className: "attention-out",
        transform: "translate(75, 310)",
      },
      title: "Out",
      value: attn1.out.toPrecision(4),
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
        title: "Key",
        value: attn2.k.toPrecision(4),
      }),

      h(SimpleWeight, {
        groupProps: {
          className: "attention-query",
          transform: "translate(125, 425)",
        },
        title: "Query",
        value: attn2.q.toPrecision(4),
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
        value: attn2.v.toPrecision(4),
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
      title: "Out",
      value: attn2.out.toPrecision(4),
    }),
    h(WeightIn, {
      groupProps: {
        transform: "translate(75, 675)",
      },
      title: "ff_net_0_proj",
      value: ff1.toPrecision(4),
    }),
    h(WeightIn, {
      groupProps: {
        transform: "translate(75, 760)",
      },
      title: "ff_net_2",
      value: ff2.toPrecision(4),
    }),

    h(WeightIn, {
      groupProps: {
        className: "cross-attention-proj-out",
        transform: "translate(75, 840)",
      },
      title: "Proj out",
      value: proj_out.toPrecision(4),
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
  return [
    h(WeightIn, { title: "conv1 (3x3)", value: conv1 }),
    h(WeightIn, { title: "time_emb_proj", value: time_emb_proj }),
    h(WeightIn, { title: "conv2 (3x3)", value: conv2 }),
    h(WeightIn, { title: "conv_shortcut", value: conv_shortcut }),
  ];
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
    d,
    ...rest,
  });
}

function LineEnd(props) {
  return h(Line, { markerEnd: "url(#head)", ...props });
}

function GText({ children, groupProps, ...rest }) {
  return h("g", groupProps, h("text", rest, children));
}

function WeightIn({ groupProps, title, value }) {
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
    h("text", { title: title, y: "2em" }, title),
    h("text", { title: title, y: "3.5em" }, value),
  );
}

function SimpleWeight({ groupProps, title, value }) {
  return h(
    "g",
    groupProps,
    h("text", { title: title }, title),
    h("text", { title: title, y: "1.5em" }, value),
  );
}

function Main({ metadata, filename }) {
  if (!metadata) {
    return h("main", null, [
      h("div", null, "No metadata for this file"),
      h(Headline, { filename }),
      h(Weight, { metadata, filename }),
      // h(Advanced, { metadata, filename }),
    ]);
  }

  return h("main", null, [
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
    // h(Advanced, { metadata, filename }),
  ]);
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

    return h(
      "div",
      { className: "full-overlay" },
      h("pre", null, JSON.stringify(sortedEntries, null, 2)),

      h(
        "div",
        { className: "action-overlay" },
        h(
          "button",
          {
            className: "download",
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
            onClick: () => {
              setShowRaw(false);
            },
          },
          "Close",
        ),
      ),
    );
  }

  return h(
    "div",
    {
      style: {
        display: "grid",
        "justify-items": "end",
        "align-items": "flex-start",
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
    raw = h(Raw, { metadata, filename });
  }

  return h("div", { className: "headline" }, [
    h("div", null, h("div", null, "LoRA file"), h("h1", null, filename)),
    raw,
  ]);
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
    h(Headline, { metadata, filename }),
    h(Header, { metadata, filename }),
    h(Main, { metadata, filename }),
  ];
}

// async function getAverageMagnitude(file) {
//   return new Promise((resolve, reject) => {
//     const reader = new FileReader();
//     reader.onload = function (e) {
//       const buffer = new Uint8Array(e.target.result);
//       const map = get_average_magnitude(buffer);
//
//       resolve(map);
//     };
//     reader.readAsArrayBuffer(file);
//   });
// }
// async function getAverageStrength(file) {
//   return new Promise((resolve, reject) => {
//     const reader = new FileReader();
//     reader.onload = function (e) {
//       const buffer = new Uint8Array(e.target.result);
//       const map = get_average_strength(buffer);
//
//       resolve(map);
//     };
//     reader.readAsArrayBuffer(file);
//   });
// }

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

let files = new Map();
let mainFilename;

// let worker = new Worker("./assets/js/worker.js", {});

const workers = new Map();

async function addWorker(file) {
  const worker = new Worker("./assets/js/worker.js", {});

  workers.set(file, worker);

  return new Promise((resolve, reject) => {
    let timeouts = [];
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

wasm_bindgen().then(() => {
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

  document
    .querySelector("#file")
    .addEventListener("change", async function (e) {
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
});

async function handleMetadata(metadata, filename) {
  dropbox.classList.remove("box__open");
  dropbox.classList.add("box__closed");
  document.querySelector("#jumbo").classList.remove("jumbo__intro");
  document.querySelector("#note").classList.add("hidden");
  const root = ReactDOM.createRoot(document.getElementById("results"));
  root.render(
    h(Metadata, {
      metadata,
      filename,
    }),
  );
}

let uploadTimeoutHandler;

async function processFile(file) {
  clearWorkers();
  const worker = await addWorker(file.name);

  terminatePreviousProcessing(file.name);

  mainFilename = undefined;

  worker.postMessage({ messageType: "file_upload", file: file });
  processingMetadata = true;
  const cancel = loading(file.name);

  uploadTimeoutHandler = setTimeout(() => {
    cancel();
    addErrorMessage("Timeout loading LoRA. Try again.");
  }, 5000);

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
        // trySyncMessage({ messageType: "keys", name: mainFilename }).then(
        //   (keys) => {
        //     console.log("keys", keys);
        //   },
        // );
        worker.postMessage({ messageType: "base_names", name: mainFilename });
        worker.postMessage({ messageType: "weight_norms", name: mainFilename });
        // worker.postMessage({ messageType: "alphas", name: mainFilename });
        // worker.postMessage({ messageType: "dims", name: mainFilename });
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
        // console.log("hasMatches", hasMatches);
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
      console.log("IS FINISHEDD!!!");
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
