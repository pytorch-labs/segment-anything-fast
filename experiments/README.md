# Running Experiments Guide

To run the experiments you need to update the script paths and install `fire`, `pandas` and `tqdm`

## Model Checkpoints

You'll need to obtain model checkpoints from the [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything) repository. Use the following commands to download them:

```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth 
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth 
```

## COCO2017 Dataset

To run experiments, you'll require the COCO2017 dataset. Download it using these commands:

```bash 
wget http://images.cocodataset.org/zips/val2017.zip 
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip 
```

## Folder Structure of Experimental Data

Here's the folder structure you should set up for your experimental data:

```plaintext
experiments_data/ 
├── tmp/ 
│   ├── sam_coco_mask_center_cache/ 
│   ├── sam_eval_masks_out/ 
├── datasets/ 
│   ├── coco2017/ 
│       ├── val2017/ 
│       ├── annotations/ 
├── checkpoints/ 
```

## Environment Details

### Hardware

These experiments were conducted on an Amazon `p4d.24xlarge` instance with the following specifications:

- 8 A100 GPUs with 40960MiB running at 400W
- 96 vCPUs
- 1152 GiB of RAM

### Software Versions

- PyTorch nightly and Python 3.10
- A fork of [facebookresearch/segment-anything](https://github.com/cpuhrsch/segment-anything) with additional commits
- [pytorch-labs/segment-anything-fast](https://github.com/pytorch-labs/segment-anything-fast)

### Installation Instructions

Follow these steps to set up the required environment:

```bash

conda create -n nightly20231117py310
conda activate nightly20231117py310
conda install python=3.10
pip install https://download.pytorch.org/whl/nightly/cu121/torch-2.2.0.dev20231117%2Bcu121-cp310-cp310-linux_x86_64.whl
pip install https://download.pytorch.org/whl/nightly/cu121/torchvision-0.17.0.dev20231117%2Bcu121-cp310-cp310-linux_x86_64.whl
git clone https://github.com/cpuhrsch/segment-anything.git
cd segment-anything
pip install -e .
cd ..
git clone https://github.com/pytorch-labs/segment-anything-fast.git
cd segment-anything-fast
pip install -e .
```

If you intend to run scripts from segment-anything-fast, install the segment-anything fork in editable mode to allow switching between different commits of the fork automatically.

### How to Run Experiments

Use this command to run experiments:

```bash
python run_experiments.py 16 vit_b <pytorch_github> <segment-anything_github> <path_to_experiments_data> --run-experiments --num-workers 32 
```

If you encounter any issues, add `--capture_output False` to increase verbosity, and feel free to open an issue.

### Data

We utilize the COCO2017 Validation (Val images) dataset for these experiments. It provides a realistic distribution of input images for measuring accuracy and performance.

### Measurement

#### Accuracy

Our primary goal is to ensure that performance optimizations do not compromise model accuracy. We do not aim to replicate paper results or make claims about model accuracy on the dataset. This measurement serves as an integration test alongside unit and other integration tests.
We calculate mask annotation center points using a simplified version of [this method](https://arxiv.org/pdf/2304.02643.pdf), section D.1.Point Sampling ([code](https://github.com/pytorch-labs/segment-anything-fast/blob/67d5c894569e99b9fdba55cfcf2f724be9f68994/experiments/data.py#L10-L120)). These points serve as annotations per image, and the number of masks and annotations per image can vary.
These images and annotations are provided to the `predict_torch` method of a `SamPredictor` instance for mask prediction. The predictions are then compared to ground truth masks using the Intersection over Union (IoU) metric ([code](https://github.com/pytorch-labs/segment-anything-fast/blob/67d5c894569e99b9fdba55cfcf2f724be9f68994/experiments/metrics.py#L4-L22)). We calculate the mean IoU (mIoU) metric over the entire 5000 images of the validation dataset to track accuracy.

#### Performance

Our objective is to measure the runtime of PyTorch models. We intentionally exclude data movement or metric calculation from measurements. Specifically, we measure the GPU execution time of running the image encoder (e.g., `vit_h`) and `SamPredictor.predict_torch` ([code](https://github.com/pytorch-labs/segment-anything-fast/blob/67d5c894569e99b9fdba55cfcf2f724be9f68994/experiments/eval_combo.py#L127-L165), [code](https://github.com/pytorch-labs/segment-anything-fast/blob/67d5c894569e99b9fdba55cfcf2f724be9f68994/experiments/eval_combo.py#L68-L99)). 
Each experiment runs in a separate Python process created from scratch. We execute three batches of warm-up before each experiment. This also means that we exclude compilation time from benchmarking. 
We measure the execution time and calculate the number of images processed per second (img/s). We also measure the maximum amount of memory allocated at the end of the process using `torch.cuda.max_memory_allocated`.

#### Tracing

We collect kernel and memory traces using PyTorch native tooling and analyze them with [Perfetto UI](https://perfetto.dev/). We typically limit the collection to a few batches to avoid generating excessively large files.

##### Kernel Traces

You can create a simple wrapper to run a function under the tracer context and write the result to a compressed JSON file. The resulting Chrome trace can be analyzed with Perfetto UI. Here's an example:

```python
def profiler_runner(path, fn, *args, **kwargs): 
    with torch.profiler.profile( 
            activities=[torch.profiler.ProfilerActivity.CPU, 
                        torch.profiler.ProfilerActivity.CUDA], 
            record_shapes=True) as prof: 
        result = fn(*args, **kwargs) 
    prof.export_chrome_trace(path) 
    return result 
```

It's useful to annotate specific regions in these traces to map code segments to the overall traces. For this, we frequently use `record_function`. See the example in the provided code.

##### Memory Profiles

We record memory history and use `memory_viz.py` to convert the result into a human-readable HTML file. Here's an example:

```python
def memory_runner(path, fn, *args, **kwargs): 
    print("Start memory recording") 
    torch.cuda.synchronize() 
    torch.cuda.memory._record_memory_history( 
        True, 
        trace_alloc_max_entries=100000, 
        trace_alloc_record_context=True 
    ) 
    result = fn(*args, **kwargs) 
    torch.cuda.synchronize() 
    snapshot = torch.cuda.memory._snapshot() 
    print("Finish memory recording") 
    import pickle 
    with open(path, 'wb') as f: 
        pickle.dump(snapshot, f) 
    # Use to convert pickle file into HTML 
    # python torch/cuda/_memory_viz.py trace_plot <snapshot>.pickle -o <snapshot>.html 
    return result 
```
