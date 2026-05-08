## PEIE: Physics Embedded Illumination Estimation for Adaptive Dehazing

### [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/32582)]

Huaizhuo Liu, [Hai-Miao Hu](https://scholar.google.com/citations?hl=zh-CN&user=ZCoORgoAAAAJ), Yonglong Jiang, Yurui Liu

> **Abstract:**  Deep learning-based methods have made significant progress in image dehazing. However, these methods often falter when applied to real-world hazy images, primarily due to the scarcity of paired real-world data and the inefficiencies of current dehazing feature extractors. Toward these issues, we introduce a novel Physics Embedded Illumination Estimation (PEIE) method for adaptive real-world dehazing. Specifically, (1) we identify the limitations of the widely used atmospheric scattering model in hazy imaging and propose a new physical model, the Illumination-Adaptive Scattering Model (IASM), for more accurate illumination representation; (2) we develop a robust data synthesis pipeline that leverages the physics embedded illumination estimation to generate realistic haze images, effectively guiding network training; and (3) we design an Illumination-Adaptive Dehazing Unit (IDU) that extracts dehazing features consistent with our proposed IASM in the latent space. By integrating the IDU into a U-Net architecture to create IADNet, we achieve significant improvements in dehazing performance through end-to-end training on synthetic data. Extensive experiments validate the superior performance of our PEIE method, significantly surpassing the state-of-the-arts in real-world dehazing.

## Environment Setup

```bash
git clone https://github.com/Hankitle/PEIE.git
cd PEIE

conda create -n PEIE python=3.8
conda activate PEIE
pip install -r requirements.txt
python setup.py develop
```

## Data and Checkpoints

- Prepare datasets and pretrained checkpoints according to paths in [options/train.yml](options/train.yml) and [options/test.yml](options/test.yml).
- Update the path fields in option files to match your local environment.

## Training

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/dist_train.sh 4 options/train.yml
```

## Testing

```bash
CUDA_VISIBLE_DEVICES=0 python basicsr/test.py -opt options/test.yml
```

## Citation

If this repo is useful for your research, please cite:

```bibtex
@inproceedings{liu2025peie,
    title={PEIE: Physics Embedded Illumination Estimation for Adaptive Dehazing},
    author={Liu, Huaizhuo and Hu, Hai-Miao and Jiang, Yonglong and Liu, Yurui},
    booktitle={AAAI},
    year={2025}
}
```

## License

Released under the [Apache 2.0 license](LICENSE).

## Acknowledgement

Built on [BasicSR](https://github.com/XPixelGroup/BasicSR). Thanks for their awesome work.

## Contact

Questions and collaborations: lhz549@buaa.edu.cn