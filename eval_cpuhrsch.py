import subprocess
import os
import math

home = "/home/cpuhrsch"

python_path = os.path.join(
    # home, "miniconda3/envs/pytorch-3.10-nightly20230727/bin/python")
    home, "miniconda3/envs/pytorch-3.10-source/bin/python")

script_path = os.path.join(
    home, "dev/segment-anything-fast/eval_combo.py")

sam_path = os.path.join(home, "dev/segment-anything")
sam_commits = {
        "default": "6fdee8f2727f4506cfbbe553e23b895e27956588",
        "graphbreaks": "55f772f77864752f2e98a6fc7713b45a1843c167",
        "codesign": "50cb459d080bcd783a4b481d3bde4150d35ac497",
        "sdpa": "22f654553bbe7aa28337ce34a25f1a9d27cee111"}

def change_sam_commit(commit_name):
    assert commit_name in sam_commits
    root_cmd = ["git", "-C", sam_path]
    result = subprocess.run(root_cmd + ["checkout", sam_commits[commit_name]], capture_output=True)
    assert result.returncode == 0
    result = subprocess.run(root_cmd + ["rev-parse", "HEAD"], capture_output=True)
    assert result.returncode == 0

root_cmd = [python_path, script_path,
            "--coco_root_dir",
            os.path.join(home, "datasets/coco2017"),
            "--coco_slice_name",
            "val2017",
            "--sam_checkpoint_base_path",
            os.path.join(home, "checkpoints/sam"),
            "--sam_model_type",
            "vit_b",
            "--point_sampling_cache_dir",
            os.path.join(home, "tmp/sam_coco_mask_center_cache"),
            "--mask_debug_out_dir",
            os.path.join(home, "tmp/sam_eval_masks_out")]


def run_experiment(idx, sam_commit_name, model_type, batch_size, num_workers, use_half=False, use_compile=False, extra_args=None, print_header=False, capture_output=True):
    change_sam_commit(sam_commit_name)
    args = root_cmd
    args = args + ["--sam_model_type", model_type]
    args = args + ["--batch_size", str(batch_size)]
    args = args + ["--num_workers", str(num_workers)]
    if use_half:
        args = args + ["--use_half", "True"]
        args = args + ["--use_half_decoder", "True"]
    if use_compile:
        args = args + ["--use_compile", "True"]
        args = args + ["--use_compile_decoder", "False"]
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
    prefix = ",".join(map(str, [idx, (t1 - t0)/60.0, sam_commit_name]))
    if result.returncode != 0:
        print(prefix + ",ERROR")
        return
    if print_header:
        header = result.stdout.decode().split("\n")[-3]
        print("idx,time,sam_commit_name," + header)
    print(prefix + "," + result.stdout.decode().split("\n")[-2])

run_experiment("018", "sdpa",        "vit_b", 60, 32, use_half=True,  use_compile=True, capture_output=False)
# run_experiment("020", "sdpa",        "vit_b", 60, 32, use_half=True,  use_compile=False,  extra_args=["--use_compile_max_autotune",  "True", "--use_compile_decoder", "False", "--use_quantize", "True"], capture_output=False)

# run_experiment("020", "sdpa",        "vit_b", 20, 32, use_half=True,  use_compile=False,  extra_args=["--use_compile_max_autotune",  "False", "--use_compile_decoder", "False", "--use_quantize", "True"], capture_output=False)
import sys; sys.exit(0)

run_experiment("010", "default",     "vit_b",  1,  0, use_half=False, use_compile=False, print_header=True)
run_experiment("011", "default",     "vit_b",  1, 32, use_half=False, use_compile=False)
run_experiment("012", "default",     "vit_b", 20, 32, use_half=False, use_compile=False)
run_experiment("013", "default",     "vit_b", 20, 32, use_half=False, use_compile=True)
run_experiment("014", "graphbreaks", "vit_b", 20, 32, use_half=False, use_compile=True)
run_experiment("015", "graphbreaks", "vit_b", 20, 32, use_half=True,  use_compile=True)
run_experiment("016", "codesign",    "vit_b", 20, 32, use_half=False, use_compile=True)
run_experiment("017", "codesign",    "vit_b", 20, 32, use_half=True,  use_compile=True)
run_experiment("018", "sdpa",        "vit_b", 20, 32, use_half=True,  use_compile=True)
run_experiment("019", "sdpa",        "vit_b", 20, 32, use_half=True,  use_compile=False,  extra_args=["--use_compile_max_autotune",  "True", "--use_compile_decoder", "False", "--use_quantize", "True", "--use_cudagraph_trees", "False"])
run_experiment("020", "sdpa",        "vit_b", 20, 32, use_half=True,  use_compile=False,  extra_args=["--use_compile_max_autotune",  "True", "--use_compile_decoder", "False", "--use_quantize", "True"])
run_experiment("021", "sdpa",        "vit_b", 20, 32, use_half=True,  use_compile=True,   extra_args=["--use_quantize", "True"])

run_experiment("110", "default",     "vit_h",  1,  0, use_half=False, use_compile=False, print_header=True)
run_experiment("111", "default",     "vit_h",  1, 32, use_half=False, use_compile=False)
run_experiment("112", "default",     "vit_h",  5, 32, use_half=False, use_compile=False)
run_experiment("113", "default",     "vit_h",  5, 32, use_half=False, use_compile=True)
run_experiment("114", "graphbreaks", "vit_h",  5, 32, use_half=False, use_compile=True)
run_experiment("115", "graphbreaks", "vit_h",  5, 32, use_half=True,  use_compile=True)
run_experiment("116", "codesign",    "vit_h",  5, 32, use_half=False, use_compile=True)
run_experiment("117", "codesign",    "vit_h",  5, 32, use_half=True,  use_compile=True)
run_experiment("118", "sdpa",        "vit_h",  5, 32, use_half=True,  use_compile=True)
run_experiment("119", "sdpa",        "vit_h",  5, 32, use_half=True,  use_compile=False,  extra_args=["--use_compile_max_autotune",  "True", "--use_compile_decoder", "False", "--use_quantize", "True", "--use_cudagraph_trees", "False"])
run_experiment("120", "sdpa",        "vit_h",  5, 32, use_half=True,  use_compile=False,  extra_args=["--use_compile_max_autotune",  "True", "--use_compile_decoder", "False", "--use_quantize", "True"])
run_experiment("121", "sdpa",        "vit_h",  5, 32, use_half=True,  use_compile=True,   extra_args=["--use_quantize", "True"])
