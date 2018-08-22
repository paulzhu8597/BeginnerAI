深度学习
===
本节包含图片的分类、识别、分割，自编码器以及各种对抗生成网络。

# 1.基础

# 2.图片分类

# 3.图片识别

# 4.图片分割

# 5.自编码器

# 6.人脸识别

# 7.对抗生成网络
## 7.1.GAN
![images](results/07_01_Z.gif)<br/>
![images](results/07_01_batman.gif)<br/>

## 7.2.DCGAN
### $28 \times 28$
| 动画 | 10 Epochs | 50 Epochs | 100 Epochs |
| ------- | ------- | ------- | ------- |
| ![images](results/07_02_DCGAN_01_MNIST.gif) | ![images](results/07_02_DCGAN_01_MNIST_10.png) | ![images](results/07_02_DCGAN_01_MNIST_50.png) | ![images](results/07_02_DCGAN_01_MNIST_100.png) |

### $32 \times 32$
| 动画 | 10 Epochs | 50 Epochs | 100 Epochs |
| ------- | ------- | ------- | ------- |
| ![images](results/07_02_DCGAN_02_Cifar10.gif) | ![images](results/07_02_DCGAN_02_Cifar10_10.png) | ![images](results/07_02_DCGAN_02_Cifar10_50.png) | ![images](results/07_02_DCGAN_02_Cifar10_100.png) |

### $64 \times 64$
| 动画 | 10 Epochs | 50 Epochs | 100 Epochs |
| ------- | ------- | ------- | ------- |
| ![images](results/07_02_DCGAN_03_Cifar10.gif) | ![images](results/07_02_DCGAN_03_Cifar10_10.png) | ![images](results/07_02_DCGAN_03_Cifar10_50.png) | ![images](results/07_02_DCGAN_03_Cifar10_100.png) |

### $96 \times 96$
| 动画 | 10 Epochs | 50 Epochs | 100 Epochs |
| ------- | ------- | ------- | ------- |
| ![images](results/07_02_DCGAN_04_AnimateFace.gif) | ![images](results/07_02_DCGAN_04_AnimateFace_10.png) | ![images](results/07_02_DCGAN_04_AnimateFace_50.png) | ![images](results/07_02_DCGAN_04_AnimateFace_100.png) |

## 7.3.CGAN
### $28 \times 28$
| 动画 | 10 Epochs | 50 Epochs | 100 Epochs |
| ------- | ------- | ------- | ------- |
| ![images](results/07_03_CGAN_01_MNIST.gif) | ![images](results/07_03_CGAN_01_MNIST_10.png) | ![images](results/07_03_CGAN_01_MNIST_50.png) | ![images](results/07_03_CGAN_01_MNIST_100.png) |

### $32 \times 32$
| 动画 | 10 Epochs | 50 Epochs | 100 Epochs |
| ------- | ------- | ------- | ------- |
| ![images](results/07_03_CGAN_02_Cifar10.gif) | ![images](results/07_03_CGAN_02_Cifar10_10.png) | ![images](results/07_03_CGAN_02_Cifar10_50.png) | ![images](results/07_03_CGAN_02_Cifar10_100.png) |

### $64 \times 64$
| 动画 | 10 Epochs | 50 Epochs | 100 Epochs |
| ------- | ------- | ------- | ------- |
| ![images](results/07_03_CGAN_03_Cifar10.gif) | ![images](results/07_03_CGAN_03_Cifar10_10.png) | ![images](results/07_03_CGAN_03_Cifar10_50.png) | ![images](results/07_03_CGAN_03_Cifar10_100.png) |

### $96 \times 96$

## 7.4.infoGAN
### $28 \times 28$
| 动画 | 1 Epoch | 30 Epochs | 50 Epochs |
| ------- | ------- | ------- | ------- |
| ![images](results/07_04_infoGAN_01_MNIST.gif) | ![images](results/07_04_infoGAN_01_MNIST_01.png) | ![images](results/07_04_infoGAN_01_MNIST_30.png) | ![images](results/07_04_infoGAN_01_MNIST_50.png) |

### $32 \times 32$

### $64 \times 64$

### $96 \times 96$

## 7.5.WGAN
### $28 \times 28$
| 动画 | 10 Epochs | 50 Epochs | 100 Epochs |
| ------- | ------- | ------- | ------- |
| ![images](results/07_05_WGAN_01_MNIST.gif) | ![images](results/07_05_WGAN_01_MNIST_10.png) | ![images](results/07_05_WGAN_01_MNIST_50.png) | ![images](results/07_05_WGAN_01_MNIST_100.png) |

### $32 \times 32$
| 动画 | 10 Epochs | 50 Epochs | 100 Epochs |
| ------- | ------- | ------- | ------- |
| ![images](results/07_05_WGAN_02_Cifar10.gif) | ![images](results/07_05_WGAN_02_Cifar10_10.png) | ![images](results/07_05_WGAN_02_Cifar10_50.png) | ![images](results/07_05_WGAN_02_Cifar10_100.png) |

### $64 \times 64$

### $96 \times 96$
| 动画 | 10 Epochs | 50 Epochs | 100 Epochs |
| ------- | ------- | ------- | ------- |
| ![images](results/07_05_WGAN_04_AnimateFace.gif) | ![images](results/07_05_WGAN_04_AnimateFace_10.png) | ![images](results/07_05_WGAN_04_AnimateFace_50.png) | ![images](results/07_05_WGAN_04_AnimateFace_100.png) |

## 7.6.WGANGP
### $28 \times 28$
| 动画 | 1 Epoch | 30 Epochs | 50 Epochs |
| ------- | ------- | ------- | ------- |
| ![images](results/07_06_WGANGP_01_MNIST.gif) | ![images](results/07_06_WGANGP_01_MNIST_01.png) | ![images](results/07_06_WGANGP_01_MNIST_30.png) | ![images](results/07_06_WGANGP_01_MNIST_50.png) |

### $32 \times 32$
| 动画 | 10 Epochs | 50 Epochs | 100 Epochs |
| ------- | ------- | ------- | ------- |
| ![images](results/07_06_WGANGP_02_Cifar10.gif) | ![images](results/07_06_WGANGP_02_Cifar10_10.png) | ![images](results/07_06_WGANGP_02_Cifar10_50.png) | ![images](results/07_06_WGANGP_02_Cifar10_100.png) |

### $64 \times 64$

### $96 \times 96$

## 7.7.LSGAN
### $28 \times 28$
| 动画 | 10 Epochs | 50 Epochs | 100 Epochs |
| ------- | ------- | ------- | ------- |
| ![images](results/07_07_LSGAN_01_MNIST.gif) | ![images](results/07_07_LSGAN_01_MNIST_10.png) | ![images](results/07_07_LSGAN_01_MNIST_50.png) | ![images](results/07_07_LSGAN_01_MNIST_50.png) |

### $32 \times 32$
| 动画 | 10 Epochs | 50 Epochs | 100 Epochs |
| ------- | ------- | ------- | ------- |
| ![images](results/07_07_LSGAN_02_Cifar10.gif) | ![images](results/07_07_LSGAN_02_Cifar10_10.png) | ![images](results/07_07_LSGAN_02_Cifar10_50.png) | ![images](results/07_07_LSGAN_02_Cifar10_100.png) |

### $64 \times 64$
| 动画 | 10 Epochs | 50 Epochs | 100 Epochs |
| ------- | ------- | ------- | ------- |
| ![images](results/07_07_LSGAN_03_Cat.gif) | ![images](results/07_07_LSGAN_03_Cat_10.png) | ![images](results/07_07_LSGAN_03_Cat_50.png) | ![images](results/07_07_LSGAN_03_Cat_100.png) |

### $96 \times 96$
| 动画 | 10 Epochs | 50 Epochs | 100 Epochs |
| ------- | ------- | ------- | ------- |
| ![images](results/07_07_LSGAN_04_AnimateFace.gif) | ![images](results/07_07_LSGAN_04_AnimateFace_10.png) | ![images](results/07_07_LSGAN_04_AnimateFace_50.png) | ![images](results/07_07_LSGAN_04_AnimateFace_100.png) |

### $128 \times 128$

## 7.8.BEGAN
### $28 \times 28$
| 动画 | 10 Epochs | 50 Epochs | 100 Epochs |
| ------- | ------- | ------- | ------- |
| ![images](results/07_08_BEGAN_01_MNIST.gif) | ![images](results/07_08_BEGAN_01_MNIST_10.png) | ![images](results/07_08_BEGAN_01_MNIST_50.png) | ![images](results/07_08_BEGAN_01_MNIST_100.png) |

### $32 \times 32$

### $64 \times 64$

### $96 \times 96$

### $128 \times 128$

## 7.9.CLSGAN
将CGAN和LSGAN相结合，效果非常不错
### $28 \times 28$
| 动画 | 10 Epochs | 50 Epochs | 100 Epochs |
| ------- | ------- | ------- | ------- |
| ![images](results/07_09_CLSGAN_01_MNIST.gif) | ![images](results/07_09_CLSGAN_01_MNIST_10.png) | ![images](results/07_09_CLSGAN_01_MNIST_50.png) | ![images](results/07_09_CLSGAN_01_MNIST_100.png) |

### $32 \times 32$
| 动画 | 10 Epochs | 50 Epochs | 100 Epochs |
| ------- | ------- | ------- | ------- |
| ![images](results/07_09_CLSGAN_02_Cifar10.gif) | ![images](results/07_09_CLSGAN_02_Cifar10_10.png) | ![images](results/07_09_CLSGAN_02_Cifar10_50.png) | ![images](results/07_09_CLSGAN_02_Cifar10_100.png) |

### $64 \times 64$

### $96 \times 96$

### $128 \times 128$

## 7.10.CBEGAN
将CGAN和BEGAN相结合
### $28 \times 28$
| 动画 | 10 Epochs | 50 Epochs | 100 Epochs |
| ------- | ------- | ------- | ------- |
| ![images](results/07_10_CBEGAN_01_MNIST.gif) | ![images](results/07_10_CBEGAN_01_MNIST_10.png) | ![images](results/07_10_CBEGAN_01_MNIST_50.png) | ![images](results/07_10_CBEGAN_01_MNIST_100.png) |

### $32 \times 32$

### $64 \times 64$

### $96 \times 96$

### $128 \times 128$
