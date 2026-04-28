# ParMIS: Parallel Maintenance of Maximal Independent Sets in Large-Scale Dynamic Graphs

ParMIS is a high-performance algorithm for maintaining **Maximal Independent Sets (MIS)** on large dynamic graphs.  
It supports both **multicore CPUs** and **many-core GPUs**, efficiently handling batches of updates with correctness guarantees and high throughput.

This repository contains CPU and GPU implementations of ParMIS, along with scripts for benchmarking against static and dynamic baselines.

---

## Features

- **Incremental MIS Updates (IMU):** Fast sequential maintenance.
- **Batch MIS Updates (BMU):** Parallel batch processing.
- **Fully Dynamic MIS (FDMU):** Handles mixed insertions and deletions.
- **Conflict-Free Region (CFR) Clustering:** Eliminates races without atomics.
- **GPU Streaming Pipeline:** Overlaps data transfer with computation for high throughput.
- Scales to **billion-edge graphs** and outperforms state-of-the-art static and dynamic MIS baselines.

---

## Repo Layout

```

ParMIS/
├── results/ # Outputs from experimental runs
├── scripts/ # Run and evaluation helper scripts
├── src/ # Core implementations (CPU & GPU)
├── README.md # This file
└── run_parmis.sh # Entry script to run ParMIS

````

---

## Datasets

We provide a **datasets.zip** archive containing several benchmark graphs used in our evaluation.

🔗 **Download:**  
https://drive.google.com/file/d/16pW0kOP31RYusLs7Nvix8iUaUC9ybg2z/view?usp=sharing

**Usage:**

1. Download and extract:

   ```bash
   wget <dataset_download_link>
   unzip datasets.zip
   ```

2. Place the extracted folder inside the `ParMIS/` directory:

   ```bash
   mv datasets/ ParMIS/
   ```

## Running the code

A driver script is provided:

```bash
./run_parmis.sh
```

---

## Output & Logs

Results and performance logs are stored in:

```
ParMIS/results/
```
