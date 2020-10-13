---
title: Docker 이미지 만들기고 배포하기
categories: 
    - Setting
toc: true
---

기존에 [deepo](https://github.com/ufoym/deepo)를 잘쓰다가 이번에 [pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning)을 써볼까 해서 기존 이미지에 새롭게 추가하려고 한다. 

막상 만들어보니 크게 어려운거 하나 없지만 내 머리는 믿을 수 없기에 나중에 다시 보기위해! 작성 시작!

# Dockerfile 란?

우선 도커 이미지를 만들려면 `Dockfile` 이라는 파일이 있어야한다. 관련 글들을 보면서 들었던 생각은 `Dockerfile` 이라는 파일이 있는 건 알겠는데.. 확장자는 뭐지..? 아니면 파일이름이 **Dockerfile** 인가? 아니면 이미지를 만들기 위해 만드는 파일을 대명사 처럼 `Dockerfile` 이라고 하는 걸까? 였다. 

정답은 그냥 파일명이 `Dockerfile` 이다. 확장자? 없다. 

만약 다른 도커 이미지들도 이어서 만들거면 도커 이미지들을 모아둘 디렉토리를 하나 만들자.

```bash
mkdir dockerimages
```

그런 다음 `Dockerfile`을 만들어서 이제 안에 이미지를 만들기 위한 내용을 담으면 된다.

```bash
vim Dockerfile
```

이제 안에 도커 이미지를 만들기위한 몇 가지 명령어를 입력해주면 된다. 명령어는 몇개 되지 않으니 크게 어려운게 없다. 오히려 너무 간단해서 도커가 더 좋을 지경이다. 

```text
FROM ufoym/deepo
MAINTAINER tootouch

RUN pip install --upgrade pip
RUN pip install tensorflow-gpu --upgrade
RUN pip install pytorch-lightning
```

Dockerfile에서 사용하는 명령어는 여러가지가 있지만 우선 내가 사용한 명령어는 **FROM**과 **RUN**이다. **MAINTAINER** 는? 할 수 있지만 이 명령어는 누가 만들었냐를 물어보는거라 명령어 뒤에 만든이 에 대한 이름을 적어주면 된다.

**FROM** 은 python을 사용한다면 쉽게 이해할 수 있다. 다른 이미지를 불러서 사용할 것이지를 말한다. 나는 **ufoym/deepo** 에 추가로 `pytorch-lightning` 만 설치할 것이기 때문에 우선 deepo를 불러온다. 

그 다음으로 `pytorch-lightning`을 설치하는데 몇 가지 추가로 업데이트를 진행해야한다. `pytorch-lightning`은 0.10.0 버전이고 `tensorboard`가 2.3.0 버전이 깔린다. 하지만 `deepo`에 있는 `tensorflow-gpu`는 2.1.0 버전이라 호환을 위해서 2.3.0 버전으로 업그레이드 해야한다. pip 명령어는 **RUN** 뒤에 입력하면 이미지를 생성할 때 함께 실행된다.

# 이미지 만들기

그리고 나서 `Dockerfile`을 저장한 후 도커의 build 명령어를 사용하면 된다. 여기서 확인할 점은 -t 옵션 뒤에는 `{이미지이름}:{태그} {Dockerfile이 있는 경로}` 이다. 

```bash
docker build -t deepo_pl:0.1 ./
```

도커 이미지가 만들어지면 `docker images` 명령어로 설치된 이미지들을 확인 할 수 있다.

```bash
REPOSITORY          TAG                 IMAGE ID            CREATED             SIZE
deepo_pl            0.1                 7ae56eb2a0bb        2 hours ago         13.9GB
```

끝! 



# Dockerfile options

추가로 `Dockerfile`에는 다른 여러 명령어도 있으니 참고자료로 남겨 놓기로 한다. 

https://docs.docker.com/engine/reference/builder/


