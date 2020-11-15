---
title: "Python으로 Notion Table 불러오기"
categories: 
    - Experiment
toc: true
---

# 인생은 노션처럼

> 인생은 "노션"을 만나기 전과 후로 나뉜다.

작년 초부터 시작해서 노션을 쓴지 1년이 넘어갑니다. 노션을 만나기 전을 회상해보자면 제 노트북은 온갖 파일들과 문서들로 지저분했었고 그마저 쓰던 애버노트와 다른 노트앱들은 얼마 쓰지 못해 간혹 들어갈때면 비밀번호도 종종 까먹던 기억이 있습니다. 그러나 노션을 처음 알게되고 사용하게되면서 정말 많은 변화가 생겼습니다. 무엇보다 무료로 수많은 문서와 파일을 넣어두고 공유와 협업마저 용이하다는게 정말 혁신이였습니다. 

최근 노션에 공부하는 자료를 입력해두고 종종 그 동안의 성적을 확인해보며 입력했던 테이블을 사용하던일이 있었는데 그때마다 collection을 압축파일로 다운받고 압출풀고 하는일이 너무 번거로웠습니다. 바로 불러오는 방법이 없을까 하다가 얼마전 알게된게 `Jamie Alexandre`가 만든 [`notion-py`](https://github.com/jamalex/notion-py)입니다. 공식 api는 아니지만 파이썬에서 바로 notion을 사용할 수 있는게 너무나 매력적이였습니다! 그러나 github을 구경하고 있었는데 아쉬웠던 것이 notion page에 새롭게 데이터를 입력하고 추가하는건 있지만 import 해오는게 없었습니다. 저에게 필요한건 테이블 자료를 불러오는 것이였기에 마침 머리도 식힐겸 `notion-py`를 이용해서 간단하게 만들어봤습니다.

# Notionist

바로 [`notionist`](https://github.com/TooTouch/notionist) ~. notion을 위한 모듈로 어떤 이름이 좋을까 하다가 이름을 notionist로 지어봤습니다. 요즘 노션 안쓰면 서러울 정도로 주변에서도 정말 많은 사람들이 노션을 쓰기 시작한 것 같습니다. 처음엔 굳이 repo로 만들이유는 없었지만 Notion으로 하고싶은 작업들이 몇 개 더 있기에 틈틈히 업로드하며 사용하려고 새로 repo를 만들었습니다. 사용 방법은 너무나 간단합니다! 

# 설치 방법

설치는 PyPI에 올려두었기에 간단하게 아래 명령어를 통해서 입력하여 사용하실 수 있습니다.

```bash
pip install notionist
```

## Token_v2

python에서 notion을 사용하기 위해서는 사용자의 token_v2가 필요합니다. 가져오는 방법은 간단합니다. 아래를 참고해주세요! 

`F12 (User Defined Key) > Application > Cookies > https://www.notion.so > token_v2`

![token_v2](https://user-images.githubusercontent.com/37654013/83939185-d2a48b80-a815-11ea-8a77-11465e01920d.JPG)


## 테이블 주소 가져오기

만약 테이블을 만들어 두신게 있으시다면 아래 사진처럼 해당 테이블의 링크를 복사할 수 있습니다. 또는 웹에서 작업중이시라면 웹 url 주소를 사용해도 됩니다.

단, 현재 모듈은 필요한 부분만 만들어져있기 때문에 특정 타입들은 가져올 수 없습니다 ㅠㅠ 

![table](https://user-images.githubusercontent.com/37654013/83939246-72fab000-a816-11ea-9894-8cd5e3d729c1.JPG)

## 자료 가져오기

이제 필요한 token_v2와 테이블 주소를 모두 가져왔으니 jupyter notebook 또는 scipt에 아래와 같이 입력해주시면 간단하게 압축파일을 다운 받을 필요없이 자료를 불러올 수 있습니다.

```python
from notionist import collection_api

token_v2 = 'YOUR token_v2'
extraction = collection_api.CollectionExtract(token_v2=token_v2)

url = 'https://www.notion.so/tootouch/ae60f9946dc54de78fbd4850ccf48b40?v=9d07e70306b2498eb82805b83f882140'
extraction.table_extract(url)
```

Tags |number|    text |Name
---|---|---|---
0    |A     | 1  | apple    |1
1    |B     | 2  |banana    |2
2    |C     | 3  |orange    |3


# 맺음말

너무나 오랜만에 잠시나마 코딩아닌 코딩을 해보니 시간가는지도 모르겠네요 ㅠㅠ 1시간 정도였지만 행복했습니다. 다음에 시간이 생기는대로 더 필요한 기능들 추가할 예정입니다. 