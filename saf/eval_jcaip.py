import subprocess
import os
import math
import itertools

home = "/home/jessecai/local"

python_path = os.path.join(
    # home, "miniconda3/envs/pytorch-3.10-nightly20230727/bin/python")
    home, "miniconda3/envs/pytorch-3.10/bin/python")

script_path = os.path.join(
    home, "segment-anything-fast/saf/eval_combo.py")

sam_path = os.path.join(home, "dev/segment-anything")
sam_commits = {
        "default": "6fdee8f2727f4506cfbbe553e23b895e27956588",
        "graphbreaks": "55f772f77864752f2e98a6fc7713b45a1843c167",
        "codesign": "50cb459d080bcd783a4b481d3bde4150d35ac497",
        "sdpa": "22f654553bbe7aa28337ce34a25f1a9d27cee111",
        "sdpa-decoder": "7dc75fdf283693f73606f2fe7fdcb693afcb16b9",
        "predict-masks-nested": "187e2359f9eb3b00d43487a1ec3db849964753e4",
        "use-rel-pos": "d2fa29d580eaf7928eef702cd71d133b943c30cf",
        "hacky-nested-encoder": "8f2fc3cc90b222a2431d4c43379282e36f021b69",
        "wip-flash-nested": "e01edb904a49c449425fca9e48902824b22cf764",
        "wip-flash-sdpa-decoder": "bb1c8b6f3749b1a5f31635f5d2f26bcafa9d94f9"}

def change_sam_commit(commit_name):
    assert commit_name in sam_commits
    root_cmd = ["git", "-C", sam_path]
    result = subprocess.run(root_cmd + ["checkout", sam_commits[commit_name]], capture_output=True)
    assert result.returncode == 0
    result = subprocess.run(root_cmd + ["rev-parse", "HEAD"], capture_output=True)
    assert result.returncode == 0

root_cmd = [python_path, script_path,
            "--coco_root_dir",
            os.path.join(home, "DATA/coco2017"),
            "--coco_slice_name",
            "val2017",
            "--sam_checkpoint_base_path",
            os.path.join(home, "MODELS"),
            "--sam_model_type",
            "vit_b",
            "--point_sampling_cache_dir",
            os.path.join(home, "tmp/sam_coco_mask_center_cache"),
            "--mask_debug_out_dir",
            os.path.join(home, "tmp/sam_eval_masks_out")]

# TODO:
# Make use_compile write out the mode
# Use max-autotune for everything
# Make epilogue fusion first a column


def run_experiment(idx,
                   sam_commit_name,
                   model_type,
                   batch_size,
                   num_workers,
                   use_half=None,
                   use_compile="False",
                   compress=None,
                   use_nested_tensor=False,
                   extra_args=None,
                   print_header=False,
                   capture_output=False,
                   limit=5,
                   profile_path=None,
                   profile_top=False,
                   memory_path=None):
    args = root_cmd
    args = args + ["--sam_model_type", model_type]
    args = args + ["--batch_size", str(batch_size)]
    args = args + ["--num_workers", str(num_workers)]
    args = args + ["--use_compile", use_compile]
    if sam_commit_name == "local-fork":
        args = args + ["--use_local_sam_fork", "True"]
    else:
        change_sam_commit(sam_commit_name)
    if use_half:
        args = args + ["--use_half", use_half]
    if compress is not None:
        args = args + ["--compress", compress]
    if use_nested_tensor:
        args = args + ["--use_nested_tensor", str(use_nested_tensor)]
    if limit is not None:
        args = args + ["--limit", str(limit)]
    if profile_path is not None:
        args = args + ["--profile-path", profile_path]
    if profile_top:
        args = args + ["--profile-top", "True"]
    if memory_path is not None:
        args = args + ["--memory-path", memory_path]
    if extra_args is None:
        extra_args = []
    args = args + extra_args
    if print_header:
        args = args + ["--print_header", "True"]
    import time
    t0 = time.time()
    result = subprocess.run(args, capture_output=capture_output)
    if not capture_output:
        return
    t1 = time.time()
    import torch
    pytorch_version = torch.__version__
    prefix = ",".join(map(str, [idx, (t1 - t0)/60.0, sam_commit_name, pytorch_version]))
    if result.returncode != 0:
        print(prefix + ",ERROR")
        return
    if print_header:
        header = result.stdout.decode().split("\n")[-3]
        print("technique,time,sam_commit_name,pytorch_version," + header)
    print(prefix + "," + result.stdout.decode().split("\n")[-2])

# TODO: Accuracy numbers for static quantization are not reliable and rely on using first 10 batches of data to build scales.
# Need to use a held out set of data to build these scalars.

def run_traces(*args, **kwargs):
    # Limit to 10 batches
    kwargs['limit'] = 160
    # Folder to save results to
    traces_dir = "/home/cpuhrsch/tmp/traces/20230924"

    # Create kernel traces
    profile_path = f"{traces_dir}/{args[0]}.json.gz"
    kwargs['profile_path'] = profile_path
    run_experiment(*args, **kwargs)
    kwargs['profile_path'] = None

    # Don't print header again if already printed
    kwargs['print_header'] = False

    # Create memory trace
    if 'use_compile' in kwargs and kwargs['use_compile'] == "max-autotune":
        # Memory traces don't seem to support CUDA graphs
        kwargs['use_compile'] = "max-autotune-no-cudagraphs"

    memory_path = f"{traces_dir}/{args[0]}"
    kwargs['memory_path'] = memory_path + ".pickle"
    run_experiment(*args, **kwargs)
    kwargs['memory_path'] = None

    # Convert memory trace to html page
    conversion_cmd = [python_path, "/home/cpuhrsch/dev/pytorch/torch/cuda/_memory_viz.py", "trace_plot", memory_path + ".pickle", "-o", memory_path + ".html"]
    result = subprocess.run(conversion_cmd, capture_output=True)
    assert result.returncode == 0

# run_traces("fp32",           "default",                     "vit_b", 16, 32, print_header=True)
# run_traces("fp16",           "codesign",                    "vit_b", 16, 32, use_half=True)
# run_traces("compile",        "codesign",                    "vit_b", 16, 32, use_half=True,  use_compile="max-autotune")
# run_traces("SDPA",           "sdpa-decoder",                "vit_b", 16, 32, use_half=True,  use_compile="max-autotune")
# run_traces("Triton",         "local-fork",                  "vit_b", 16, 32, use_half=True,  use_compile="max-autotune")
# run_traces("NT",             "local-fork",                  "vit_b", 16, 32, use_half=True,  use_compile="max-autotune", use_nested_tensor=True)
# run_traces("int8",           "local-fork",                  "vit_b", 16, 32, use_half=True,  use_compile="max-autotune", use_nested_tensor=True, compress="dynamic_quant")
# run_traces("sparse",         "local-fork",                  "vit_b", 16, 32, use_half=True,  use_compile="max-autotune", use_nested_tensor=True, compress="int4_dynamic_quant_sparse")

print_header = True
# for bs, model in itertools.product([1, 32, 64], ["vit_b", "vit_h"]):
for bs, model in itertools.product([32, 64], ["vit_b", "vit_h"]):
    run_experiment("sparse",      "local-fork",   f"{model}_2x4_wanda", bs, 32, use_half="bfloat16",  use_compile="max-autotune", use_nested_tensor=(bs > 1), compress="sparse")

import sys; sys.exit(0)
    # run_experiment("fp32",        "default",                     model, bs, 32, print_header=print_header, capture_output=False)
    # print_header = False
    # run_experiment("bf16",        "codesign",                    model, bs, 32, use_half="bfloat16")

    # run_experiment("compile",     "codesign",                    model, bs, 32, use_half="bfloat16",  use_compile="max-autotune")
    # run_experiment("SDPA",        "sdpa-decoder",                model, bs, 32, use_half="bfloat16",  use_compile="max-autotune")
    # run_experiment("Triton",      "local-fork",                  model, bs, 32, use_half="bfloat16",  use_compile="max-autotune")
    # if bs > 1:
    #     run_experiment("NT",      "local-fork",                  model, bs, 32, use_half="bfloat16",  use_compile="max-autotune", use_nested_tensor=(bs > 1))
    # run_experiment("int8",        "local-fork",                  model, bs, 32, use_half="bfloat16",  use_compile="max-autotune", use_nested_tensor=(bs > 1), compress="dynamic_quant")
    # run_experiment("sparse",      "local-fork",   f"{model}_2x4_wanda", bs, 32, use_half="bfloat16",  use_compile="max-autotune", use_nested_tensor=(bs > 1), compress="sparse")
    # run_experiment("sparse_int8", "local-fork",                  model, bs, 32, use_half="bfloat16",  use_compile="max-autotune", use_nested_tensor=(bs > 1), compress="int4_dynamic_quant_sparse")
