# response.py GitHub bundle

`response.py`를 실행하는 데 필요한 코드와 설정만 따로 모아둔 폴더입니다.

포함한 것:
- `response.py`
- `src/` 하위의 실제 실행 의존 모듈
- `configs/version.json`
- 실행용 `requirements.txt`

의도적으로 제외한 것:
- `outputs/` 결과물
- `data/` 원본 문서와 구조화 결과
- `.deps_*` 로컬 패키지 폴더
- 테스트, 실험, UI 보조 스크립트

## 실행 방법

```powershell
cd response_github_bundle
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python response.py --host 127.0.0.1 --port 8233
```

브라우저에서 `http://127.0.0.1:8233` 로 열면 됩니다.

## OpenAI 설정

OpenAI 요약/QA 기능까지 쓰려면 `.env.example`을 `.env`로 복사해서 값을 넣으면 됩니다.

```env
OPENAI_API_KEY=your_key
OPENAI_MODEL=gpt-5.2
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

`.env`가 없어도 서버는 뜨지만, OpenAI 기반 요약/응답은 비활성화되거나 제한됩니다.

## 참고

- 현재 번들은 "코드만" 정리한 것이어서 기존 프로젝트의 문서 데이터와 산출물은 포함하지 않았습니다.
- 업로드한 문서를 처리하면서 필요한 `outputs/` 폴더는 실행 중 자동 생성됩니다.
- LangChain reranker까지 완전히 쓰려면 환경에 따라 `sentence-transformers` 추가 설치가 필요할 수 있습니다.
