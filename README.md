## Reproducing Neural Discrete Representation Learning
### Course Project for [IFT 6135 - Representation Learning](https://ift6135h18.wordpress.com/)

Project Report link: [final_project.pdf](final_project.pdf)

### Instructions
1. To train the VQVAE with default arguments as discussed in the report, execute:
```
python vqvae.py --data-folder /tmp/miniimagenet --output-folder models/vqvae
python vqvae.py --data-folder cifar-10-batches-py --dataset cifar10 --output-folder models --device cuda
python vqvae.py --ckp models/models/best.pt --lr 1e-5 --batch-size 256 --num-epochs 200 --output-folder models --device cuda --data-folder cifar-10-batches-py --dataset cifar10 --tmodel H:\pytorch-cifar\checkpoint\ckpt.pth
```
2. To train the PixelCNN prior on the latents, execute:
```
python pixelcnn_prior.py --data-folder /tmp/miniimagenet --model models/vqvae --output-folder models/pixelcnn_prior
python pixelcnn_prior.py --data-folder cifar-10-batches-py --dataset cifar10 --model models/best.pt --output-folder models/pixelcnn_prior
```
### Datasets Tested
#### Image
1. MNIST
2. FashionMNIST
3. CIFAR10
4. Mini-ImageNet

#### Video
1. Atari 2600 - Boxing (OpenAI Gym) [code](https://github.com/ritheshkumar95/pytorch-vqvae/tree/evan/video)

### Reconstructions from VQ-VAE
Top 4 rows are Original Images. Bottom 4 rows are Reconstructions.
#### MNIST
![png](samples/vqvae_reconstructions_MNIST.png)
#### Fashion MNIST
![png](samples/vqvae_reconstructions_FashionMNIST.png)

### Class-conditional samples from VQVAE with PixelCNN prior on the latents
#### MNIST
![png](samples/samples_MNIST.png)
#### Fashion MNIST
![png](samples/samples_FashionMNIST.png)

### Comments
1. We noticed that implementing our own VectorQuantization PyTorch function speeded-up training of VQ-VAE by nearly 3x. The slower, but simpler code is in this [commit](https://github.com/ritheshkumar95/pytorch-vqvae/tree/cde142670f701e783f29e9c815f390fc502532e8).
2. We added some basic tests for the vector quantization functions (based on `pytest`). To run these tests
```
py.test . -vv
```

### Authors
1. Rithesh Kumar
2. Tristan Deleu
3. Evan Racah
