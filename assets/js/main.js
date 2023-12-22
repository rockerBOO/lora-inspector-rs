// import init, {
//   // get_metadata,
//   // get_average_magnitude,
//   // get_average_strength,
//   // get_average_magnitude_by_block,
//   // get_average_strength_by_block,
// } from "/pkg/lora_inspector_rs.js";
//
// const { startup } = wasm_bindgen;

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
  const [alphas, setAlphas] = React.useState([
    metadata.get("ss_network_alpha"),
  ]);
  const [dims, setDims] = React.useState([metadata.get("ss_network_dim")]);
  const [networkModule, setNetworkModule] = React.useState(
    metadata.get("ss_network_module"),
  );
  const [networkType, setNetworkType] = React.useState("");
  const [networkArgs, setNetworkArgs] = React.useState(
    metadata.has("ss_network_args")
      ? JSON.parse(metadata.get("ss_network_args"))
      : null,
  );

  // TODO: date parsing isn't working right or the date is invalid
  // const trainingStart = new Date(Number.parseInt(metadata.get("ss_training_started_at"))).toLocaleString()
  // const trainingEnded = new Date(Number.parseInt(metadata.get("ss_training_ended_at"))).toLocaleString()

  React.useEffect(() => {
    trySyncMessage({ messageType: "alphas", name: mainFilename }).then(
      (resp) => {
        console.log("ALPHAS", resp);
        setAlphas(resp.alphas);
      },
    );
    trySyncMessage({ messageType: "dims", name: mainFilename }).then((resp) => {
      console.log("DIMS", resp);
      setDims(resp.dims);
    });
    trySyncMessage({ messageType: "network_args", name: mainFilename }).then(
      (resp) => {
        setNetworkArgs(resp.networkArgs);
      },
    );
    trySyncMessage({ messageType: "network_module", name: mainFilename }).then(
      (resp) => {
        setNetworkModule(resp.networkModule);
      },
    );
    trySyncMessage({ messageType: "network_type", name: mainFilename }).then(
      (resp) => {
        setNetworkType(resp.networkType);
      },
    );
  }, []);

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
      h(MetaAttribute, {
        name: "Network Rank/Dimension",
        valueClassName: "rank",
        value: dims.join(", "),
      }),
      h(MetaAttribute, {
        name: "Network Alpha",
        valueClassName: "alpha",
        value: alphas.join(", "),
      }),
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
  ];
  // h("div", {}, [
  // h("div", { title: "seed" }, metadata.get("ss_seed")),
  //    h(
  //      "div",
  //      { title: "Training started at" },
  //      trainingStart,
  //    ),
  //    h(
  //      "div",
  //      { title: "Training ended at" },
  // trainingEnded
  //    ),
  // ]),
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
  // const [averageStrength, setAverageStrength] = React.useState(undefined);
  // const [averageMagnitude, setAverageMagnitude] = React.useState(undefined);

  // React.useEffect(() => {
  //   setAverageStrength(get_average_strength(buffer));
  //   setAverageMagnitude(get_average_magnitude(buffer));
  // }, []);

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
      // h(MetaAttribute, {
      //   name: "Average vector strength, UNet + TE",
      //   valueClassName: "number",
      //   value: averageStrength?.toPrecision(4),
      // }),
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

function Blocks({ metadata, filename }) {
  // console.log("!!!! BLOCKS !!!!! METADATA FILENAME", filename);
  const [hasBlockWeights, setHasBlockWeights] = React.useState(false);
  const [teMagBlocks, setTEMagBlocks] = React.useState(new Map());
  const [unetMagBlocks, setUnetMagBlocks] = React.useState(new Map());
  const [teStrBlocks, setTEStrBlocks] = React.useState(new Map());
  const [unetStrBlocks, setUnetStrBlocks] = React.useState(new Map());
  const [normProgress, setNormProgress] = React.useState(0);
  const [currentCount, setCurrentCount] = React.useState(0);
  const [totalCount, setTotalCount] = React.useState(0);
  const [startTime, setStartTime] = React.useState(undefined);
  const [currentBaseName, setCurrentBaseName] = React.useState("");

  const teChartRef = React.useRef(null);
  const unetChartRef = React.useRef(null);

  React.useEffect(() => {
    if (!hasBlockWeights) {
      return;
    }

    // const averageMagnitudes = get_average_magnitude_by_block(buffer);
    // const averageStrength = get_average_strength_by_block(buffer);

    console.log("getting l2 norms...");
    trySyncMessage({
      messageType: "l2_norm",
      name: filename,
      reply: true,
    }).then((resp) => {
      console.log(resp);
      // setTEMagBlocks(averageMagnitudes.get("text_encoder"));
      setTEMagBlocks(resp.norms.te);
      setUnetMagBlocks(resp.norms.unet);
      //
      // setTEStrBlocks(averageStrength.get("text_encoder"));
      // setUnetStrBlocks(averageStrength.get("unet"));
    });

    return function cleanup() {};
  }, [hasBlockWeights]);

  React.useEffect(() => {
    if (!hasBlockWeights) {
      return;
    }

    setStartTime(performance.now());

    listenProgress("l2_norms_progress").then(async (getProgress) => {
      while ((progress = await getProgress().next())) {
        const value = progress.value;
        setCurrentBaseName(value.baseName);
        setCurrentCount(value.currentCount);
        setTotalCount(value.totalCount);
        setNormProgress(value.currentCount / value.totalCount);
      }
    });

    return function cleanup() {};
  }, [hasBlockWeights]);

  React.useEffect(() => {
    if (!teChartRef.current && !unetChartRef.current) {
      return;
    }

    const makeChart = (dataset, chartRef, strBlocks) => {
      const data = {
        // A labels array that can contain any sort of values
        labels: dataset.map(([k, _]) => k),
        // Our series array that contains series objects or in this case series data arrays
        series: [
          dataset.map(([_k, v]) => v["mean"]),
          // dataset.map(([k, v]) => strBlocks.get(k)),
        ],
      };
      console.log("chartdata", data);
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
      makeChart(Array.from(teMagBlocks), teChartRef, teStrBlocks);
    }
    if (unetMagBlocks.size > 0) {
      makeChart(Array.from(unetMagBlocks), unetChartRef, unetStrBlocks);
    }
  }, [teMagBlocks, teStrBlocks, unetMagBlocks, unetStrBlocks]);

  // console.log(teMagBlocks, unetMagBlocks);

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
      h("h3", {}, "Text Encoder Block Weights"),
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
                name: `${k} norm ${v["metadata"]["type"]}`,
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
      h("h3", {}, "UNet Block Weights"),
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
              name: `${k} l2 norm  ${v["metadata"]["type"]}`,
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
    h(MetaAttribute, {
      name: "Gradient Checkpointing",
      value: metadata.get("ss_gradient_checkpointing"),
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
  ]);
}

function Loss({ metadata }) {
  return h("div", { className: "row space-apart" }, [
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
    h(MetaAttribute, {
      name: "Masked Loss",
      value: metadata.get("ss_masked_loss"),
    }),
  ]);
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
  return [
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
  ];
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
  return Object.entries(tagFrequency)
    .sort((a, b) => a[1] < b[1])
    .map(([tag, count], i) => {
      const alt = i % 2 > 0 ? " alt-row" : "";
      return h(
        "div",
        { className: "tag-frequency" + alt },
        h("div", {}, count),
        h("div", {}, tag),
      );
    });
}

function Main({ metadata, filename }) {
  if (!metadata) {
    return h("main", null, [h(Weight, { metadata, filename })]);
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
    h(Dataset, { metadata }),
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
    return h("main", {}, h(NoMetadata, { filename }));
  }

  return [
    h(Headline, { metadata, filename }),
    h(Header, { metadata, filename }),
    // h("div", {}, "wtf"),
    // h(Blocks, { metadata, buffer }),
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

let worker = new Worker("./assets/js/worker.js", {});
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
  console.log("HANDLE METADATA", filename);
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
  terminatePreviousProcessing();

  mainFilename = undefined;
  worker.postMessage({ messageType: "file_upload", file: file });
  processingMetadata = true;
  const cancel = loading();

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

      console.log(":) :) MAIN FILENAME :) :) ", mainFilename);

      handleMetadata(e.data.metadata, file.name).then(() => {
        worker.postMessage({
          messageType: "network_module",
          name: mainFilename,
        });
        worker.postMessage({ messageType: "network_args", name: mainFilename });
        worker.postMessage({ messageType: "network_type", name: mainFilename });
        worker.postMessage({ messageType: "weight_keys", name: mainFilename });
        worker.postMessage({ messageType: "alpha_keys", name: mainFilename });
        worker.postMessage({ messageType: "keys", name: mainFilename });
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
function terminatePreviousProcessing() {
  if (processingMetadata) {
    // restart the worker
    worker.terminate();
    // make a new worker
    worker = new Worker("./assets/js/worker.js");
  }

  processingMetadata = false;
}

function cancelLoading() {
  terminatePreviousProcessing();
  finishLoading();
  clearTimeout(uploadTimeoutHandler);
}

window.addEventListener("keyup", (e) => {
  if (e.key === "Escape") {
    cancelLoading();
  }
});

function loading() {
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
      cancelLoading();
    }
  });

  return function cancel() {
    cancelLoading();
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

async function trySyncMessage(message) {
  return new Promise((resolve) => {
    worker.postMessage({ ...message, reply: true });

    const workerHandler = (e) => {
      if (e.data.messageType === message.messageType) {
        worker.removeEventListener("message", workerHandler);
        resolve(e.data);
      }
    };

    console.log("Attaching sync message for ", message);

    worker.addEventListener("message", workerHandler);
  });
}

async function listenProgress(messageType) {
  let isFinished = false;
  function finishedWorkerHandler(e) {
    if (e.data.messageType === `${messageType}_finished`) {
      worker.removeEventListener("message", finishedWorkerHandler);
      isFinished = true;
    } else {
      // console.log("unhandled finished message", e.data);
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

// async function getBlockWeights() {
//   return ;
// }
