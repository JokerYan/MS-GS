# Multi-Scale 3D Gaussian Splatting for Anti-Aliased Rendering
Zhiwen Yan, Weng Fei Low, Yu Chen, Gim Hee Lee <br>
[Project Page](https://jokeryan.github.io/projects/ms-gs/) | [Paper](https://arxiv.org/abs/2311.17089) 

### Release Note
I would like to apologize to all researchers interested in this paper.
We should have released the instruction for training and evaluating the repo much earlier.
If there is any issue with the instruction or the code, 
please feel free to drop me an email at `yan.zhiwen@u.nus.edu`.

Thank you for your support and we welcome you to cite our paper at
```json
@inproceedings{yan2024multi,
  title={Multi-scale 3d gaussian splatting for anti-aliased rendering},
  author={Yan, Zhiwen and Low, Weng Fei and Chen, Yu and Lee, Gim Hee},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={20923--20931},
  year={2024}
}
```

### Environment
The environment requirement largely follows the requirement of the original 3DGS.
For the step of rasterization module installation step, 
please install the rasterization module of this paper using `pip install submodules/diff-gaussian-rasterization/` instead.

Please note that this might replace the vanilla rasterization module of 3DGS, 
so we suggest you install everything in a fresh environment.

### Training
```bash
python train.py
    --eval                  // evaluate the model
    -s ${scene directory, contains images and sparse}
    -m ${model output directory, ./output/train/ms_test}
    --ms_train              // use multi-scale training
    --ms_train_max_scale 6  // maximum scale for multi-scale training
    --test_iterations 1000
    --test_interval 2500
    --iterations 40000
    --filter_small          // filter small gs
    --insert_large          // insert large gs
```

### Viewing
We wrote a simple viewer to help visualizing the output model.
```bash
python viewer.py
    -s ${scene directory, contains images and sparse}
    -m ${model output directory, ./output/train/ms_test}
    --anti_alias
```

You can use:
* `x` and `c` key to switch to different test scenes. 
* `7` and `9` to move forward and backward in the scene.
* `4` and `6` to move left and right
* `.` and `/` to adjust view resolution scale
* `[` and `]` to adjust the Gaussian size
* `z` to reset the view
* `q` to quit
* 's' to save the current view to an image
