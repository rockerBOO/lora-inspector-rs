import init, {
  get_metadata,
  get_average_magnitude,
  get_average_strength,
  get_average_magnitude_by_block,
  get_average_strength_by_block,
} from "./pkg/lora_inspector_rs.js";

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
    null,
    "model spec",
    h("div", { className: "row space-apart" }, [
      h("div", { title: "Date" }, "Date: " + metadata.get("modelspec.date")),
      h("div", { title: "Title" }, metadata.get("modelspec.title")),
      h(
        "div",
        { title: "Prediction" },
        metadata.get("modelspec.prediction_type"),
      ),
    ]),
    h("div", { className: "row space-apart" }, [
      h(
        "div",
        { title: "License" },
        "License: " + metadata.get("modelspec.license"),
      ),
      h("div", { title: "Description" }, metadata.get("modelspec.description")),
    ]),
    h("div", { className: "row space-apart" }, [
      h("div", { title: "Tags" }, metadata.get("modelspec.tags")),
    ]),
  );
}

function PretrainedModel({ metadata }) {
  return h(
    "div",
    { className: "row space-apart" }, //
    h(MetaAttribute, { className: "caption" }, "SD Model"),
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

function Network({ metadata }) {
  const networkArgs = metadata.has("ss_network_args")
    ? JSON.stringify(JSON.parse(metadata.get("ss_network_args")))
    : "";

  // TODO: date parsing isn't working right or the date is invalid
  // const trainingStart = new Date(Number.parseInt(metadata.get("ss_training_started_at"))).toLocaleString()
  // const trainingEnded = new Date(Number.parseInt(metadata.get("ss_training_ended_at"))).toLocaleString()

  return [
    h(
      "div",
      { className: "row space-apart" }, //
      h(MetaAttribute, {
        name: "Network module",
        value: metadata.get("ss_network_module"),
      }),
      h(MetaAttribute, {
        name: "Network Rank/Dimension",
        valueClassName: "rank",
        value: metadata.get("ss_network_dim"),
      }),
      h(MetaAttribute, {
        name: "Network Alpha",
        valueClassName: "alpha",
        value: metadata.get("ss_network_alpha"),
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
        value: networkArgs,
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

  return h(
    "div",
    null,
    h(MetaAttribute, { name: "LR Scheduler", value: lrScheduler }),
    h(MetaAttribute, {
      name: "LR Scheduler arguments",
      valueClassName: "args",
      value: metadata.get("ss_lr_scheduler_args"),
    }),
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
  );
}

function Optimizer({ metadata }) {
  return h("div", null, [
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

function Weight({ metadata, buffer }) {
  const [averageStrength, setAverageStrength] = React.useState(undefined);
  const [averageMagnitude, setAverageMagnitude] = React.useState(undefined);

  React.useEffect(() => {
    setAverageStrength(get_average_strength(buffer));
    setAverageMagnitude(get_average_magnitude(buffer));
  }, []);

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
        name: "Average vector strength, UNet + TE",
        valueClassName: "number",
        value: averageStrength?.toPrecision(4),
      }),
      h(MetaAttribute, {
        name: "Average vector magnitude, UNet + TE",
        valueClassName: "number",
        value: averageMagnitude?.toPrecision(4),
      }),
    ]),
    h(Blocks, { metadata, buffer }),
  ];
}

function Blocks({ metadata, buffer }) {
  const [hasBlockWeights, setHasBlockWeights] = React.useState(false);
  const [teMagBlocks, setTEMagBlocks] = React.useState(new Map());
  const [unetMagBlocks, setUnetMagBlocks] = React.useState(new Map());
  const [teStrBlocks, setTEStrBlocks] = React.useState(new Map());
  const [unetStrBlocks, setUnetStrBlocks] = React.useState(new Map());
  React.useEffect(() => {
    if (!hasBlockWeights) {
      return;
    }

    const averageMagnitudes = get_average_magnitude_by_block(buffer);
    const averageStrength = get_average_strength_by_block(buffer);

    setTEMagBlocks(averageMagnitudes.get("text_encoder"));
    setUnetMagBlocks(averageMagnitudes.get("unet"));

    setTEStrBlocks(averageStrength.get("text_encoder"));
    setUnetStrBlocks(averageStrength.get("unet"));
    return function cleanup() {};
  }, [hasBlockWeights]);

  if (!hasBlockWeights) {
    return h(
      "button",
      {
        className: "primary",
        onClick: (e) => {
          e.preventDefault();
          console.log("get block weights");
          setHasBlockWeights((state) => (state ? false : true));
        },
      },
      "Get block weights",
    );
  }

  return [
    h("h3", {}, "Text Encoder Block Weights"),
    h(
      "div",
      { className: "block-weights text-encoder" },
      Array.from(teMagBlocks)
        .sort(([a, _], [b, _v]) => a > b)
        .map(([k, v]) => {
          // console.log("te-block", k, v);
          return h(
            "div",
            null,
            h(MetaAttribute, {
              name: `${k} average strength`,
              value: teStrBlocks.get(k).toPrecision(4),
              valueClassName: "number",
            }),
            h(MetaAttribute, {
              className: "te-block",
              name: `${k} average magnitude`,
              value: v.toPrecision(4),
              valueClassName: "number",
            }),
          );
        }),
    ),
    h("h3", {}, "UNet Block Weights"),
    h(
      "div",
      { className: "block-weights unet" },
      Array.from(unetMagBlocks)
        .sort(([a, _], [b, _v]) => a > b)
        .map(([k, v]) => {
          // console.log("unet-block", k, v, unetStrBlocks.get(k));
          return h(
            "div",
            null,
            h(MetaAttribute, {
              className: "unet-block",
              name: `${k} average magnitude`,
              value: v.toPrecision(4),
              valueClassName: "number",
            }),
            h(MetaAttribute, {
              name: `${k} average strength`,
              value: unetStrBlocks.get(k).toPrecision(4),
              valueClassName: "number",
            }),
          );
        }),
    ),
  ];
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
    h("div", {}, "Buckets:"),
    h(
      "div",
      { className: "row space-apart" },
      h(BucketInfo, { metadata, dataset }),
    ),
    h(
      "div",
      { className: "row space-apart" },

      "Subsets:",
    ),
    h(
      "div",
      { className: "subsets" },
      dataset["subsets"].map((subset) => h(Subset, { metadata, subset })),
    ),

    h(
      "div",
      { className: "row space-apart" },
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
      return h(MetaAttribute, {
        name: key,
        value: `${bucket["resolution"][0]}x${bucket["resolution"][1]}: ${bucket["count"]}`,
      });
    }),
  ];
}

function Subset({ subset, metadata }) {
  return h("div", { className: "subset" }, [
    h(MetaAttribute, { name: "Class Token", value: subset["class_tokens"] }),
    h(MetaAttribute, { name: "Color aug", value: subset["color_aug"] }),
    h(MetaAttribute, { name: "Flip aug", value: subset["flip_aug"] }),
    h(MetaAttribute, { name: "image dir", value: subset["image_dir"] }),
    h(MetaAttribute, { name: "image count", value: subset["img_count"] }),
    h(MetaAttribute, { name: "is_reg", value: subset["is_reg"] }),
    h(MetaAttribute, { name: "keep tokens", value: subset["keep_tokens"] }),
    h(MetaAttribute, { name: "num repeats", value: subset["num_repeats"] }),
    h(MetaAttribute, {
      name: "shuffle caption",
      value: subset["shuffle_caption"],
    }),
  ]);
}

function TagFrequency({ tagFrequency, metadata }) {
  return Object.entries(tagFrequency).map(([tag, count]) => {
    return h("div", {}, `${tag}: ${count}`);
  });
}

function Main({ metadata, buffer }) {
  return h("main", null, [
    h(PretrainedModel, { metadata }),
    h(Network, { metadata }),
    h(LRScheduler, { metadata }),
    h(Optimizer, { metadata }),
    h(Weight, { metadata, buffer }),
    h(EpochStep, { metadata }),
    h(Batch, { metadata }),
    h(Noise, { metadata }),
    h(Loss, { metadata }),
    h(Dataset, { metadata }),
  ]);
}

function Metadata({ metadata, buffer }) {
  return [
    h(Header, { metadata }),
    // h("div", {}, "wtf"),
    // h(Blocks, { metadata, buffer }),
    h(Main, { metadata, buffer }),
  ];
}

async function readMetadata(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = function (e) {
      const buffer = new Uint8Array(e.target.result);
      const metadata = get_metadata(buffer);

      resolve([metadata, buffer]);
    };
    reader.readAsArrayBuffer(file);
  });
}
async function getAverageMagnitude(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = function (e) {
      const buffer = new Uint8Array(e.target.result);
      const map = get_average_magnitude(buffer);

      resolve(map);
    };
    reader.readAsArrayBuffer(file);
  });
}
async function getAverageStrength(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = function (e) {
      const buffer = new Uint8Array(e.target.result);
      const map = get_average_strength(buffer);

      resolve(map);
    };
    reader.readAsArrayBuffer(file);
  });
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

init().then(() => {
  ["drop"].forEach((evtName) => {
    document.addEventListener(evtName, (e) => {
      e.preventDefault();
      e.stopPropagation();

      const droppedFiles = e.dataTransfer.files;

      const results = [];
      for (let i = 0; i < droppedFiles.length; i++) {
        results.push(readMetadata(droppedFiles.item(i)));
        // getAverageMagnitude(droppedFiles.item(i)).then((v) => {
        //   console.log("average_magnitude", v);
        // });
        // getAverageStrength(droppedFiles.item(i)).then((v) => {
        //   console.log("average_strength", v);
        // });
      }
      handleFile(results);
    });
  });

  document.querySelector("#file").addEventListener("change", (e) => {
    e.preventDefault();
    e.stopPropagation();

    const files = e.target.files;

    const results = [];
    for (let i = 0; i < files.length; i++) {
      results.push(readMetadata(files.item(i)));
      // getAverageMagnitude(files.item(i)).then((v) => {
      //   console.log("average_magnitude", v);
      // });
      // getAverageStrength(files.item(i)).then((v) => {
      //   console.log("average_strength", v);
      // });
    }
    handleFile(results);
  });

  async function handleFile(results) {
    const metadatas = await Promise.all(results);

    dropbox.classList.remove("box__open");
    dropbox.classList.add("box__closed");
    document.querySelector("#jumbo").classList.remove("jumbo__intro");
    document.querySelector("#note").classList.add("hidden");
    metadatas.forEach(([metadata, buffer]) => {
      const root = ReactDOM.createRoot(document.getElementById("results"));
      root.render(
        h(Metadata, {
          metadata,
          buffer,
        }),
      );
    });
  }
});
