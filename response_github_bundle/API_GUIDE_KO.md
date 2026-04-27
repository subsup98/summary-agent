# API 가이드: 이 코드베이스에서 API를 찾고 직접 추가하는 방법

이 문서는 지금 프로젝트 기준으로 `API를 어디서 정의하는지`, `프론트엔드가 어디서 호출하는지`, 그리고 `직접 새 API를 만드는 순서`를 차근차근 정리한 가이드입니다.

초보자 기준으로 보면 이 프로젝트는 크게 3층으로 나뉩니다.

1. `Python 서버`
   `response_github_bundle/main.py`가 `/api/...` 경로를 실제로 처리합니다.
2. `브라우저 UI`
   `response_github_bundle/src/ui/static/document-studio.js`가 `fetch()`로 서버 API를 호출합니다.
3. `외부 API 호출`
   서버 내부에서 OpenAI 같은 외부 API를 직접 부르는 코드는 `src/retrieval/...`, `src/indexing/...` 쪽에 있습니다.

## 1. 지금 코드에서 API를 "정의"하는 곳

가장 먼저 볼 파일은 [main.py](/C:/Users/yongseop.im/Desktop/summary_agent/response_github_bundle/main.py:273) 입니다.

여기서 중요한 포인트는 3개입니다.

### 1-1. OpenAPI 문서 목록

`build_openapi_spec()` 함수에서 현재 서버가 제공하는 API 목록을 문서처럼 정리합니다.

예:

```python
def build_openapi_spec(host: str, port: int) -> dict[str, object]:
    return {
        "paths": {
            "/api/documents": {"get": {...}},
            "/api/run": {"post": {...}},
            "/api/summarize": {"post": {...}},
            "/api/query": {"post": {...}},
        }
    }
```

이 부분은 "실제 동작" 자체라기보다, `우리 서버에 이런 API가 있다`는 설명서 역할에 가깝습니다.

### 1-2. 요청 라우팅

[main.py](/C:/Users/yongseop.im/Desktop/summary_agent/response_github_bundle/main.py:2785) 의 `do_GET()` / `do_POST()`에서 URL 경로를 보고 어떤 함수를 실행할지 결정합니다.

실제 코드 흐름은 이런 식입니다.

```python
def do_GET(self) -> None:
    path = urlparse(self.path).path
    if path == "/api/documents":
        self._send_json({"documents": self.manager.get_document_list()})
        return

def do_POST(self) -> None:
    path = urlparse(self.path).path
    if path == "/api/summarize":
        self._handle_summarize()
        return
    if path == "/api/query":
        self._handle_query()
        return
```

즉, 브라우저가 `/api/query`로 POST 요청을 보내면, 서버는 `_handle_query()`를 실행합니다.

### 1-3. 실제 처리 함수

예를 들어 [main.py](/C:/Users/yongseop.im/Desktop/summary_agent/response_github_bundle/main.py:3476) 의 `_handle_query()`는 이렇게 동작합니다.

1. 요청 본문(JSON)을 읽습니다.
2. `query`, `strategy`, `document_id` 같은 값을 꺼냅니다.
3. `self.retriever.answer_question(...)`에 넘깁니다.
4. 결과를 JSON으로 돌려줍니다.

핵심 부분만 보면:

```python
def _handle_query(self) -> None:
    payload = self._read_json_payload()
    query = str(payload.get("query", "")).strip()
    if not query:
        self._send_json({"error": "query_is_required"}, status=400)
        return

    answer = self.retriever.answer_question(
        query=query,
        strategy=strategy,
        document_id=document_id,
        source_name=source_name,
        document_ids=selected_document_ids,
    )
    self._send_json(answer, status=200)
```

이게 이 프로젝트의 대표적인 "API 서버 코드"입니다.

## 2. 지금 코드에서 API를 "호출"하는 곳

프론트엔드 호출은 [document-studio.js](/C:/Users/yongseop.im/Desktop/summary_agent/response_github_bundle/src/ui/static/document-studio.js:986) 를 보면 됩니다.

### 문서 목록 조회

```javascript
const response = await fetch("/api/documents", { cache: "no-store" });
const payload = await response.json();
state.documents = payload.documents || [];
```

### 업로드 실행

```javascript
const response = await fetch("/api/run", { method: "POST", body: formData });
const payload = await response.json();
```

### 질문 보내기

```javascript
const response = await fetch("/api/query", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    query,
    strategy: defaultQaStrategy,
    document_id: useScopedOnly ? "" : selected.document_id,
    source_name: useScopedOnly ? "" : selected.source_name,
    selected_document_ids: scopedDocumentIds,
  }),
});
const payload = await response.json();
```

즉, 브라우저의 JS는 `fetch("/api/...")`를 호출하고, 서버의 `main.py`는 그 요청을 받아 `_handle_...()` 함수에서 처리합니다.

## 3. 이 프로젝트 안에 "외부 API 호출"도 있는가?

있습니다. 그것도 명확하게 존재합니다.

대표 파일:

- [openai_answerer.py](/C:/Users/yongseop.im/Desktop/summary_agent/response_github_bundle/src/retrieval/openai_answerer.py:14)
- [document_summary.py](/C:/Users/yongseop.im/Desktop/summary_agent/response_github_bundle/src/retrieval/document_summary.py:343)
- [embedding_backends.py](/C:/Users/yongseop.im/Desktop/summary_agent/response_github_bundle/src/indexing/embedding_backends.py:22)
- [retrieval_qa.py](/C:/Users/yongseop.im/Desktop/summary_agent/response_github_bundle/src/retreival_lanchain/retrieval_qa.py:854)

예를 들어 `openai_answerer.py`는 OpenAI Responses API를 직접 부릅니다.

```python
OPENAI_API_URL = "https://api.openai.com/v1/responses"

req = request.Request(
    OPENAI_API_URL,
    data=json.dumps(payload).encode("utf-8"),
    headers={
        "Authorization": f"Bearer {self.settings.api_key}",
        "Content-Type": "application/json",
    },
    method="POST",
)
```

이건 "우리 서비스의 내부 API"가 아니라, `우리 서버가 OpenAI 서버에 보내는 외부 API 호출`입니다.

정리하면:

- `main.py`의 `/api/query`, `/api/summarize` 같은 것은 우리가 제공하는 내부 API
- `src/retrieval/openai_answerer.py` 같은 것은 외부 서비스(OpenAI)를 호출하는 코드

## 4. 요청이 실제로 흘러가는 방식

`질문하기` 기능을 예로 들면 흐름은 아래와 같습니다.

1. 사용자가 브라우저 UI에서 질문 입력
2. `document-studio.js`가 `fetch("/api/query", ...)` 호출
3. `main.py`의 `do_POST()`가 `/api/query`를 보고 `_handle_query()` 실행
4. `_handle_query()`가 `self.retriever.answer_question(...)` 호출
5. 필요하면 내부에서 검색, 요약, 외부 LLM 호출 진행
6. 서버가 JSON 응답 반환
7. 브라우저 JS가 `response.json()`으로 받아 화면에 렌더링

이 흐름을 이해하면 새 API를 추가할 때도 똑같이 만들 수 있습니다.

## 5. 새 API를 만들 때 기본 순서

이 프로젝트에서는 아래 순서로 작업하면 가장 헷갈리지 않습니다.

1. `무슨 URL`로 받을지 정합니다.
   예: `/api/hello`
2. `GET인지 POST인지` 정합니다.
   조회성은 보통 GET, 데이터 전달은 POST
3. `main.py`의 `do_GET()` 또는 `do_POST()`에 경로를 연결합니다.
4. `main.py`에 `_handle_hello()` 같은 처리 함수를 만듭니다.
5. 필요하면 `build_openapi_spec()`에도 문서 항목을 추가합니다.
6. 브라우저에서 쓸 거면 `document-studio.js`에서 `fetch("/api/hello")`를 호출합니다.
7. 응답 JSON 구조를 단순하게 유지합니다.

## 6. 가장 쉬운 연습: GET API 하나 만들기

처음에는 데이터베이스나 OpenAI 연결 없이, 아주 간단한 GET API부터 만드는 게 좋습니다.

예시 목표:

- 경로: `/api/hello`
- 메서드: `GET`
- 응답: `{"message": "hello", "guide": "api practice"}`

### Step 1. OpenAPI 문서에 추가

`build_openapi_spec()`의 `paths`에 아래 항목을 추가합니다.

```python
"/api/hello": {
    "get": {
        "tags": ["system"],
        "summary": "Simple practice endpoint",
        "responses": {
            "200": {
                "description": "Practice response",
                "content": {
                    "application/json": {
                        "schema": {"type": "object", "additionalProperties": True}
                    }
                },
            }
        },
    }
},
```

### Step 2. 라우팅에 연결

`do_GET()`에 아래 코드를 추가합니다.

```python
if path == "/api/hello":
    self._handle_hello()
    return
```

### Step 3. 처리 함수 만들기

`main.py` 안에 아래 함수를 추가합니다.

```python
def _handle_hello(self) -> None:
    self._send_json(
        {
            "message": "hello",
            "guide": "api practice",
        },
        status=200,
    )
```

이 3개만 하면 서버 API 하나가 완성됩니다.

### Step 4. 브라우저 또는 JS에서 호출

브라우저 개발자 도구 콘솔이나 JS 코드에서 이렇게 부르면 됩니다.

```javascript
const response = await fetch("/api/hello");
const payload = await response.json();
console.log(payload);
```

예상 응답:

```json
{
  "message": "hello",
  "guide": "api practice"
}
```

## 7. 그다음 연습: POST API 하나 만들기

실전에서는 보통 POST가 더 자주 필요합니다. 사용자가 값을 보내면 서버가 받아서 응답하는 형태입니다.

예시 목표:

- 경로: `/api/echo`
- 메서드: `POST`
- 요청: `{"name": "Yong"}`
- 응답: `{"message": "안녕하세요, Yong"}`

### Step 1. `do_POST()`에 연결

```python
if path == "/api/echo":
    self._handle_echo()
    return
```

### Step 2. 처리 함수 만들기

```python
def _handle_echo(self) -> None:
    payload = self._read_json_payload()
    name = str(payload.get("name", "")).strip()
    if not name:
        self._send_json({"error": "name_is_required"}, status=400)
        return

    self._send_json({"message": f"안녕하세요, {name}"}, status=200)
```

### Step 3. 프론트에서 호출

```javascript
const response = await fetch("/api/echo", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ name: "Yong" }),
});
const payload = await response.json();
console.log(payload);
```

## 8. 이 프로젝트 스타일에 맞게 API를 만들 때 체크할 것

### 8-1. 입력 검증 먼저 하기

이 코드베이스는 없는 값이 들어오면 바로 400을 반환하는 패턴을 많이 씁니다.

예:

```python
if not query:
    self._send_json({"error": "query_is_required"}, status=400)
    return
```

새 API를 만들 때도 이 스타일을 그대로 따라가면 일관성이 좋아집니다.

### 8-2. 성공/실패 응답 구조를 단순하게 유지하기

처음에는 아래 정도만 지켜도 충분합니다.

- 성공: 필요한 데이터만 반환
- 실패: `{"error": "..."}` 형태 유지

예:

```python
self._send_json({"result": result}, status=200)
self._send_json({"error": "invalid_input"}, status=400)
```

### 8-3. GET과 POST를 구분하기

- GET: 조회
- POST: 생성, 실행, 계산, 업로드, 질문 제출

현재 코드도 이 규칙을 대체로 따릅니다.

- `GET /api/documents`
- `POST /api/run`
- `POST /api/summarize`
- `POST /api/query`

### 8-4. 프론트와 서버의 JSON 키 이름을 맞추기

예를 들어 프론트에서 이렇게 보내면:

```javascript
body: JSON.stringify({ query, strategy })
```

서버도 정확히 같은 키로 읽어야 합니다.

```python
query = str(payload.get("query", "")).strip()
strategy = str(payload.get("strategy", DEFAULT_QA_STRATEGY)).strip()
```

이 부분이 API 초반 실수의 대부분입니다.

## 9. 외부 API 호출을 직접 만들고 싶다면

이 프로젝트에는 이미 좋은 참고 예제가 있습니다.

대표적으로 [openai_answerer.py](/C:/Users/yongseop.im/Desktop/summary_agent/response_github_bundle/src/retrieval/openai_answerer.py:55) 를 보면 다음 순서로 작성합니다.

1. `.env` 또는 환경변수에서 API 키 읽기
2. 요청 payload 만들기
3. 헤더에 `Authorization: Bearer ...` 넣기
4. HTTP POST 요청 보내기
5. 응답 JSON 파싱하기
6. 실패 시 예외 처리하기

### 가장 단순한 외부 API 호출 예시

```python
import json
from urllib import request


def call_example_api(api_key: str) -> dict:
    payload = {
        "model": "gpt-5.2",
        "input": "안녕하세요",
    }
    req = request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with request.urlopen(req, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))
```

이 코드는 "브라우저에서 OpenAI를 직접 호출"하는 것이 아니라, `우리 Python 서버가 OpenAI를 대신 호출`하는 구조입니다.

보통은 이 방식이 더 안전합니다.

## 10. 내가 직접 따라 해보는 추천 순서

처음부터 큰 기능을 만들기보다 아래 순서를 추천합니다.

1. `GET /api/hello` 만들기
   서버 라우팅과 JSON 응답에 익숙해지기 좋습니다.
2. `POST /api/echo` 만들기
   JSON 요청 읽는 법을 익히기 좋습니다.
3. `document-studio.js`에서 버튼이나 함수 하나로 호출해보기
   프론트와 서버 연결 감을 익힐 수 있습니다.
4. 그다음에 실제 기능 API 만들기
   예: 문서 메타데이터 조회, 특정 문서 통계, 간단한 검색
5. 마지막으로 외부 OpenAI API 연결
   키 관리, 예외 처리, 응답 파싱까지 들어갑니다.

## 11. 이 프로젝트에서 특히 먼저 읽으면 좋은 파일

- [main.py](/C:/Users/yongseop.im/Desktop/summary_agent/response_github_bundle/main.py:273)
  API 목록, 라우팅, 핸들러가 한 파일에 모여 있습니다.
- [document-studio.js](/C:/Users/yongseop.im/Desktop/summary_agent/response_github_bundle/src/ui/static/document-studio.js:986)
  프론트에서 API를 어떻게 호출하는지 바로 보입니다.
- [openai_answerer.py](/C:/Users/yongseop.im/Desktop/summary_agent/response_github_bundle/src/retrieval/openai_answerer.py:55)
  외부 OpenAI API 호출 예시로 가장 읽기 쉽습니다.
- [embedding_backends.py](/C:/Users/yongseop.im/Desktop/summary_agent/response_github_bundle/src/indexing/embedding_backends.py:62)
  임베딩 API 호출 패턴을 볼 수 있습니다.

## 12. 마지막 요약

이 프로젝트에는 이미 API 작성과 호출 코드가 모두 있습니다.

- 내부 API 작성: `main.py`
- 내부 API 호출: `src/ui/static/document-studio.js`
- 외부 OpenAI API 호출: `src/retrieval/*.py`, `src/indexing/*.py`

가장 좋은 학습 방법은:

1. 기존 `/api/query` 흐름을 읽어보기
2. `GET /api/hello`를 직접 추가해보기
3. `POST /api/echo`를 만들어보기
4. 프론트 `fetch()`와 연결해보기

원하시면 다음 단계로 바로 이어서 도와드릴 수 있습니다.

- 제가 `연습용 /api/hello` 엔드포인트를 실제 코드에 추가해드리기
- 버튼까지 붙여서 브라우저에서 눌러보게 만들기
- 또는 `OpenAI 호출용 새 API`를 하나 같이 만들기
