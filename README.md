# An Official Code Implementation of FUNCTO: Function-Centric One-Shot Imitation Learning for Tool Manipulation
[[paper]](https://arxiv.org/abs/2502.11744) [[website]](https://sites.google.com/view/functo) [[video]](https://www.youtube.com/watch?v=E_NXAZKRvWk&t=39s)

[Chao Tang](https://mkt1412.github.io/), Anxing Xiao, Yuhong Deng, Tianrun Hu, Wenlong Dong, Hanbo Zhang, David Hsu, and Hong Zhang  

If you find this work useful, please cite:
```
@article{tang2025functo,
  title={Functo: Function-centric one-shot imitation learning for tool manipulation},
  author={Tang, Chao and Xiao, Anxing and Deng, Yuhong and Hu, Tianrun and Dong, Wenlong and Zhang, Hanbo and Hsu, David and Zhang, Hong},
  journal={arXiv preprint arXiv:2502.11744},
  year={2025}
}
  ```

## Installation

Please begin by installing the following dependency packages: Open3D, SciPy, PyTorch, and CasADi. 

**Our code also relies on OWL-ViT, Grounding-SAM, and SD-DINO (optional), which are currently deployed on our internal servers. To use these models locally, please follow the installation instructions provided below.**

1) Installing [[OWL-ViT]](https://huggingface.co/docs/transformers/en/model_doc/owlv2) or another object detector of your choice.

2) Installing [[Grounding-SAM]](https://github.com/IDEA-Research/Grounded-Segment-Anything) or another segmentation model of your choice.

3) Installing [[SD+DINO]](https://github.com/Junyi42/sd-dino) (optional).

4) After installing these models, please replace the code blocks in the original code that were used to call them from our internal servers.

## Demo 
**We provide a demo showcasing the task of pouring:**

1) Specify the parameters in the `config.yaml` file located under `utils_IL/config`. The `vp_flag` parameter controls whether to use a VLM for pose alignment refinement, while `sd_dino_flag` determines whether to use SD+DINO for functional keypoint transfer.
Both parameters are set to False by default.

2) To run the demo:
```
python main.py
```

3) Check the generated trajectory at `test_data/pour_test/tool_traj_transfer_output/test_tool_traj_pc.ply`.



