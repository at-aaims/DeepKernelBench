# Wan2.1
-----

## Introduction of Wan2.1

**Wan2.1**  is designed on the mainstream diffusion transformer paradigm, achieving significant advancements in generative capabilities through a series of innovations. These include our novel spatio-temporal variational autoencoder (VAE), scalable training strategies, large-scale data construction, and automated evaluation metrics. Collectively, these contributions enhance the model’s performance and versatility.

## Quickstart

Install dependencies:
```sh
# Ensure torch >= 2.4.0
pip install -r requirements.txt
```

Run the program on a system with NVIDIA or AMD GPUs
```sh
python main.py
```

Run the program on a system with Intel GPUs
```sh
python main_xpu.py
```

After the program executes successfully, the generated video is `output.mp4`


> 💡Note: 
> * The 1.3B model is capable of generating videos at 720P resolution. However, due to limited training at this resolution, the results are generally less stable compared to 480P. For optimal performance, we recommend using 480P resolution. 
> * The 1.3B model requires ~23 GB of GPU memory

> 💡Troubleshooting: 
To address the MIOpen Error: /longer_pathname_so_that_rpms_can_support_packaging_the_debug_info_for_all_os_profiles/src/rocm-libraries/projects/miopen/src/include/miopen/kern_db.hpp:181: Internal error while accessing SQLite database: attempt to write a readonly database
```sh
export MIOPEN_USER_DB_PATH="/tmp/my-miopen-cache"
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
rm -rf ${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}
```
