# TextVR
 A Large Cross-Modal Video Retrieval Dataset with Reading Comprehension

## Get Started
 ### Environment Setup
 TextVR is based on the video retrieval model [
frozen-in-time](https://github.com/m-bain/frozen-in-time). To setup the environment of TextVR, we use `conda` to manage our dependencies. Our developers use `CUDA 11.1` to do experiments. You can specify the appropriate `cudatoolkit` version to install on your machine in the `requirements.txt` file, and then run the following commands to install FlowText:
 ```
conda create -n textvr python=3.8
conda activate textvr

pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

git clone https://github.com/callsys/TextVR
cd TextVR
pip install -r requirements.txt
 ```
### Download Dataset and Model
To run TextVR, you need to download some files, which mainly contain the dataset, the pretrained model. Once you have downloaded the files, link them to the TextVR directory:
```
ln -s path/to/TextVR_data TextVR/data
```
### Running TextVR
Training TextVR with given config file `configs/TextVR_fusion.json`:
```
python train.py -c configs/TextVR_fusion.json
```
Inference TextVR with given weights `ckpt/textvr.pth` and save the similarity matrix `textvr.npy`:
```
python test.py -c configs/TextVR_fusion.json -r ckpt/textvr.pth --sim_path textvr.npy
```

```
python gen.py --video assets/demo.mp4 --range 0,400,5 --save assets/result --seed 16
```
