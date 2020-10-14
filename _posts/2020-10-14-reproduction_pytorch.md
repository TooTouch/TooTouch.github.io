---
title: 파이토치 모델 결과 재구성하기 (Pytorch Reproduction Experiement)
categories: 
    - Setting
toc: true
---

연구나 공모전을 하면서 여러 실험을 반복하다 보면 같은 실수를 반복하기 마련이다. 그중에 가장 아쉬운 실수는 실컷 모델 실험해서 제출했더니 나중에 구현이 안 돼서 난감한 경우가 많았다. 

이번 기회에 random seed를 고정하는 방법들과 실제로 적용했을 때 똑같은 결과가 나오는지 확인하기 위해 간단히 요약했다. Pytorch 환경에서 randomness를 고정하기 위한 방법은 이미 잘 정리된 [블로그](https://hoya012.github.io/blog/reproducible_pytorch/)가 있었으니... 간단하게 결과만 쓰기로 한다.

# Methods

모델 학습 시 랜덤한 값을 갖는 경우는 여러 경우가 있지만, 대표적으로 랜덤하게 할당하는 초기 가중치와 augmentation에 사용되는 random 값이 대표적이다. 이를 위해 부분별로 필요한 randomness를 제어할 수 있는 기능이 필요하다.

## Random Seed Code

각 부분별로 randomness를 제어할 수 있는 코드를 하나로 모았다. 아래 코드를 요약하자면 간단하다. 하지만 이걸 다 실험하면서 찾아주신 분들께는 감사드린다는 말씀을 드리면서.. 간단 요약!

- Pytorch의 random seed 고정
- Python의 random으로 pytorch transforms의 random seed 고정
- CUDA와 CuDNN random seed 고정

```python
import torch
import numpy as np 
import random

def torch_seed(random_seed=201014):

    torch.manual_seed(random_seed)

    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(random_seed)
    random.seed(random_seed)

```

## 실험 코드

간단한 실험을 위한 코드는 `pytorch-lightning`을 활용했고 Tensorboard로 시각화해서 비교했다. 숫자로 보는 게 더 정확하지만 그림으로 보는 게 더 편해서... 생략!

우선 실험을 위한 코드 구성! 

`pytorch-lightning`은 checkpoint와 tensorboard를 **lightning_logs**라는 파일을 만들어서 버전별로 관리할 수 있다. model.py는 아래 [부록](http://tootouch.github.io/setting/reproduction_pytorch/#부록)에 첨부하였다. 

```bash
.
├── MNIST
├── lightning_logs
├── main.py
└── model.py
```

main.py은 아래와 같이 구성되어 있는데 실험은 random seed를 어디에 고정하는 게 좋을지 실험했다. 당연히 첫 시작에 위치하는 게 가장 좋겠지만..!(?) 지금 생각해보니 그렇지만 그냥 비교해봤다.

1. 모델 생성
2. 학습기(trainer) 설정
3. 학습 시작

```python
from model import LitMNIST, seed_everything
import pytorch_lightning as pl

# 모델 생성
model = LitMNIST(hidden_size=64, learning_rate=0.0001)

# trainer 설정
trainer = pl.Trainer(
    gpus=1, 
    max_epochs=30, 
    progress_bar_refresh_rate=20
)

# 학습 시작
trainer.fit(model)
```

실험을 위한 파이프 라인은 위의 main.py에 random seed 위치에 따라 크게 세 가지로 나누었다. 그리고 재구현이 되는지를 확인하기 위해 각각 두 번씩 반복해서 모델을 학습했다.

1. **Random seed가 없는 상태** (version 0, version 1)
2. **학습기 설정 전** (version 2, version 3)
3. **모델 생성 전** (version 4, version 5)

학습에 사용한 설정값은 다음과 같다.

```python
BATCH_SIZE = 64
EPOCH = 30
LEARNING_RATE = 0.0001
```

# Result

결과는 tensorboard dev로 업로드한 링크에서 확인할 수 있다. 

- [실험 결과 한 번에 보기](https://tensorboard.dev/experiment/X7NncAhBQcG30feR09sN4Q/#scalars&runSelectionState=eyJsaWdodG5pbmdfbG9ncy92ZXJzaW9uXzAiOmZhbHNlLCJsaWdodG5pbmdfbG9ncy92ZXJzaW9uXzEiOmZhbHNlLCJsaWdodG5pbmdfbG9ncy92ZXJzaW9uXzIiOmZhbHNlLCJsaWdodG5pbmdfbG9ncy92ZXJzaW9uXzMiOmZhbHNlLCJsaWdodG5pbmdfbG9ncy92ZXJzaW9uXzQiOnRydWUsImxpZ2h0bmluZ19sb2dzL3ZlcnNpb25fNSI6dHJ1ZX0%3D&_smoothingWeight=0.605)!



## Random seed가 없는 상태

첫 번째로 random seed를 설정하지 않은 상태에서 비교한 두 번의 학습 로그이다. Epoch 단위로 비교했을 때는 꽤 비슷해 보일 수 있지만, 실제 값을 보면 차이가 있음을 알 수 있다.  

<p align='center'>
    <img src='https://user-images.githubusercontent.com/37654013/95955200-6b3bbf80-0e37-11eb-84c9-03de8aabde7b.gif'>
</p>

하지만 역시 batch 단위로 비교했을 때 확인한 결과 상단한 차이가 있음을 알 수 있다. 

<p align='center'>
    <img src='https://user-images.githubusercontent.com/37654013/95957799-26198c80-0e3b-11eb-87e5-7307d31a3420.gif'>
</p>


## 학습기 생성 전 

두 번째로 random seed를 모델 생성 후 그리고 학습기 설정 전 사이에서 고정한 수 학습을 하였다. 그 결과 역시 epoch 단위로 보았을 때 첫 번째와 마찬가지로 거의 똑같아 보이지만 실제로 그 값의 차이가 있다. 

<p align='center'>
    <img src='https://user-images.githubusercontent.com/37654013/95957404-a55a9080-0e3a-11eb-901c-1c6fd88bdea4.gif'>
</p>

첫 번째와 다르다고 한다면 비교적 batch 단위에서 그 차이가 줄어들었음을 알 수 있다. 그 이유는 ... 무엇일까..? CUDA와 CuDNN의 randomness가 고정되어서 때문일까도 싶다. 

<p align='center'>
    <img src='https://user-images.githubusercontent.com/37654013/95957816-2b76d700-0e3b-11eb-8e26-3729a5a516cc.gif'>
</p>


## 모델 생성 전 

마지막으로 모델 생성 전에 random seed를 고정한 후 학습을 한 결과이다. Epoch 단위로 보았을 때 똑같은 값이 그래도 재현되었음을 알 수 있다. 

<p align='center'>
    <img src='https://user-images.githubusercontent.com/37654013/95957535-cb803080-0e3a-11eb-9ff3-a7b37824d0a4.gif'>
</p>

Batch 단위로 보았을 때도 정확히 일치함을 알 수 있다. 

<p align='center'>
    <img src='https://user-images.githubusercontent.com/37654013/95957819-2ca80400-0e3b-11eb-843d-18b0afcdd2f2.gif'>
</p>


# 결과

실험적으로 확인했을 때 **seed 고정은 모델 생전 바로 전**이 가장 적합해 보인다. 모델 생성 전이 아닌 최상단에 seed를 고정하는 경우 실험 과정 중 모델 생성 전 코드 실행에서 중간에 추가되는 환경 변화의 차이가 있을 수 있기 때문에 seed 고정은 모델 생성 전으로 하는 것이 최선이라고 생각한다. 

기본적으로 위에서 구성한 torch_seed 함수 정도면 어느 정도 모델은 결과를 재구성하는 데 충분히 도움이 될 거라 생각된다. 하지만 MNIST를 가지고 간단한 분류 문제로 실험해본 결과라서 간단하게 모델 결과를 재구성해볼 수 있었지만, 모델과 문제가 복잡할수록 다른 문제가 있을 수도 있다.

# 부록

**실험코드**

실험에서 pytorch-lightning 을 사용해봤는데 굉장히 편리했지만 뭔가 아직은 찝찝한 것들이 있긴하다. 확실히 모든 것에 편할 수는 없나보다. model.py의 코드는 pytorch-lightning에서 제공하는 [example code](https://colab.research.google.com/github/PytorchLightning/pytorch-lightning/blob/master/notebooks/01-mnist-hello-world.ipynb)를 활용했고 수정한 부분은 tensorboard에 train과 validation의 로그를 출력하는 부분 정도이다.

이번에 처음 pytorch-lightning을 사용해봤는데 굉장히 편리하다. LightningModule 클래스를 상속해서 모델 클래스를 만들 수 있는데 아직 써보지 못했다면 한 번쯤 구조를 확인해봐도 좋을 것 같아서 첨부한다.

**model.py**

```python
import os
import random
import numpy as np

import torch
from torch import nn 
import torch.nn.functional as F 

from torchvision.datasets import MNIST 
from torch.utils.data import DataLoader, random_split 
from torchvision import transforms 

import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy


def seed_everything(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    

class LitMNIST(pl.LightningModule):
    
    def __init__(self, data_dir='./', hidden_size=64, learning_rate=2e-4):

        super().__init__()

        # Set our init args as class attributes
        self.data_dir = data_dir
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        # Hardcode some dataset specific attributes
        self.num_classes = 10
        self.dims = (1, 28, 28)
        channels, width, height = self.dims
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

        # Define PyTorch model
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * width * height, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.num_classes)
        )

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):

        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        tensorboard_logs = {'train_loss': loss, 'train_acc': acc}

        return {'loss': loss, 'acc': acc, 'log':tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['acc'] for x in outputs]).mean()

        self.logger.experiment.add_scalar("Loss/Train",
                                            avg_loss,
                                            self.current_epoch)

        self.logger.experiment.add_scalar("Accuracy/Train",
                                            avg_acc,
                                            self.current_epoch)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        tensorboard_logs = {'val_loss': loss, 'val_acc': acc}

        '''
        아래 코드를 사용하면 progress bar에서 validation에 대한 결과를 볼 수 있지만 tensorboard에 자동으로 기록되는 단점이 있다.
        왜 단점이냐면, epoch별 마지막 batch에 대한 결과만 기록되는데 전체 epoch에 대한 값이 아니라 의미가 없다. 
        
        따라서 위의 아래 validation_epoch_end 를 추가해서 epoch 단위로 로그를 기록했다. 

        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        self.log('val_acc', acc, prog_bar=True, on_epoch=True)
        '''

        return {'val_loss': loss, 'val_acc': acc, 'log':tensorboard_logs}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()

        self.logger.experiment.add_scalar("Loss/Val",
                                            avg_loss,
                                            self.current_epoch)

        self.logger.experiment.add_scalar("Accuracy/Val",
                                            avg_acc,
                                            self.current_epoch)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    ####################
    # DATA RELATED HOOKS
    ####################

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=32)

```
