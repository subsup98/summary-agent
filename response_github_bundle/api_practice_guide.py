"""
api_practice_guide.py

이 파일은 "이 프로젝트에서 API를 어떻게 작성하는지"를 연습하기 위한
독립 실행형 예제 파일입니다.

중요:
- 이 파일은 기존 main.py에 자동으로 연결되지 않습니다.
- 즉, 학습용/실습용 예제 파일입니다.
- 현재 프로젝트 스타일을 최대한 닮게 만들었습니다.

이 파일에서 보여주는 것:
1. GET API 작성 예시:    GET  /api/hello
2. POST API 작성 예시:   POST /api/echo
3. 상태 조회 API 예시:   GET  /api/status
4. 연습용 추가 API:
   - GET  /api/document-count
   - POST /api/document-exists
   - POST /api/add-numbers
5. 외부 API 호출 함수 예시: call_openai_responses_example()

이 파일을 읽으면 아래 흐름을 이해할 수 있습니다.

1. URL 경로를 어떻게 분기하는가
2. GET과 POST를 어떻게 나누는가
3. JSON 요청 본문을 어떻게 읽는가
4. JSON 응답을 어떻게 반환하는가
5. 잘못된 요청은 왜 400으로 처리하는가
6. 예외는 왜 500으로 처리하는가
7. 외부 API 호출 함수는 어떤 모양으로 작성하는가

실행 예시:
    cd response_github_bundle
    python api_practice_guide.py

브라우저에서 열기:
    http://127.0.0.1:8260/
    http://127.0.0.1:8260/docs
    http://127.0.0.1:8260/openapi.json

PowerShell에서 POST 테스트:
    Invoke-RestMethod -Uri "http://127.0.0.1:8260/api/echo" `
      -Method POST `
      -ContentType "application/json" `
      -Body '{"name":"Yong"}'
"""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib.parse import urlparse


# 이 파일은 "실습용 서버"이므로 포트를 고정해서 단순하게 보여줍니다.
# 실제 서비스에서는 argparse나 설정 파일로 받을 수 있습니다.
HOST = "127.0.0.1"
PORT = 8260


def build_openapi_like_spec() -> dict[str, Any]:
    """
    현재 프로젝트의 build_openapi_spec() 아이디어를 단순화한 함수입니다.

    왜 이런 문서 함수를 두는가?
    - 서버가 어떤 API를 제공하는지 사람이 한눈에 파악하기 쉽습니다.
    - Swagger/OpenAPI UI로 연결할 때 기반 데이터가 됩니다.
    - 프론트엔드/백엔드가 같은 계약(contract)을 공유하기 좋습니다.

    여기서는 학습용이므로 완전한 OpenAPI가 아니라
    "이 API가 어떤 역할을 하는지"만 보여주는 간단한 dict를 반환합니다.
    """
    return {
        "openapi": "3.0.3",
        "info": {
            "title": "Practice API Guide",
            "version": "1.0.0",
            "description": "Standalone practice server for learning API routing, request handling, frontend integration, and Swagger testing.",
        },
        "servers": [{"url": f"http://{HOST}:{PORT}"}],
        "tags": [
            {"name": "system", "description": "Health and simple GET practice endpoints"},
            {"name": "practice", "description": "Small practice APIs for GET/POST exercises"},
        ],
        "paths": {
            "/api/status": {
                "get": {
                    "tags": ["system"],
                    "summary": "Get server status",
                    "responses": {
                        "200": {
                            "description": "Server status",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/StatusResponse"}
                                }
                            },
                        }
                    },
                }
            },
            "/api/hello": {
                "get": {
                    "tags": ["practice"],
                    "summary": "Simple GET practice endpoint",
                    "responses": {
                        "200": {
                            "description": "Hello response",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/HelloResponse"}
                                }
                            },
                        }
                    },
                }
            },
            "/api/document-count": {
                "get": {
                    "tags": ["practice"],
                    "summary": "Get the number of sample documents",
                    "responses": {
                        "200": {
                            "description": "Document count response",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/DocumentCountResponse"}
                                }
                            },
                        }
                    },
                }
            },
            "/api/echo": {
                "post": {
                    "tags": ["practice"],
                    "summary": "Echo back a provided name",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/EchoRequest"}
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Echo response",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/EchoResponse"}
                                }
                            },
                        },
                        "400": {"$ref": "#/components/responses/ErrorResponse"},
                    },
                }
            },
            "/api/document-exists": {
                "post": {
                    "tags": ["practice"],
                    "summary": "Check whether a document_id exists in sample data",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/DocumentExistsRequest"}
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Exists check response",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/DocumentExistsResponse"}
                                }
                            },
                        },
                        "400": {"$ref": "#/components/responses/ErrorResponse"},
                    },
                }
            },
            "/api/add-numbers": {
                "post": {
                    "tags": ["practice"],
                    "summary": "Add two numbers",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/AddNumbersRequest"}
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Addition result",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/AddNumbersResponse"}
                                }
                            },
                        },
                        "400": {"$ref": "#/components/responses/ErrorResponse"},
                    },
                }
            },
        },
        "components": {
            "schemas": {
                "StatusResponse": {
                    "type": "object",
                    "properties": {
                        "ok": {"type": "boolean"},
                        "service": {"type": "string"},
                        "version": {"type": "string"},
                    },
                    "required": ["ok", "service", "version"],
                },
                "HelloResponse": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string"},
                        "guide": {"type": "string"},
                    },
                    "required": ["message", "guide"],
                },
                "DocumentCountResponse": {
                    "type": "object",
                    "properties": {
                        "count": {"type": "integer"},
                        "documents": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["count", "documents"],
                },
                "EchoRequest": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
                "EchoResponse": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string"},
                        "received": {
                            "type": "object",
                            "properties": {"name": {"type": "string"}},
                            "required": ["name"],
                        },
                    },
                    "required": ["message", "received"],
                },
                "DocumentExistsRequest": {
                    "type": "object",
                    "properties": {"document_id": {"type": "string"}},
                    "required": ["document_id"],
                },
                "DocumentExistsResponse": {
                    "type": "object",
                    "properties": {
                        "document_id": {"type": "string"},
                        "exists": {"type": "boolean"},
                    },
                    "required": ["document_id", "exists"],
                },
                "AddNumbersRequest": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "number"},
                        "b": {"type": "number"},
                    },
                    "required": ["a", "b"],
                },
                "AddNumbersResponse": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "number"},
                        "b": {"type": "number"},
                        "sum": {"type": "number"},
                    },
                    "required": ["a", "b", "sum"],
                },
                "ErrorResponseSchema": {
                    "type": "object",
                    "properties": {
                        "error": {"type": "string"},
                        "detail": {"type": "string"},
                        "path": {"type": "string"},
                    },
                },
            },
            "responses": {
                "ErrorResponse": {
                    "description": "Error payload",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/ErrorResponseSchema"}
                        }
                    },
                }
            },
        },
    }


def render_swagger_ui_html() -> str:
    """
    간단한 Swagger UI 페이지입니다.

    참고:
    - Swagger는 "필수"는 아닙니다.
    - 프론트 버튼으로 정상 호출이 되면 실제 연동 확인은 충분히 가능합니다.
    - 다만 Swagger가 있으면 요청 스펙과 응답 형태를 문서처럼 확인하고
      Try it out으로 빠르게 테스트하기 좋습니다.

    이 페이지는 CDN에서 Swagger UI 자산을 불러옵니다.
    따라서 인터넷이 막혀 있으면 스타일이나 스크립트가 로드되지 않을 수 있습니다.
    그 경우에도 /openapi.json 은 열어볼 수 있습니다.
    """
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Practice API Swagger</title>
  <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css">
  <style>
    html, body { margin: 0; padding: 0; background: #f8fafc; }
    body { font-family: "Segoe UI", Arial, sans-serif; }
    .topbar { display: none; }
    .offline-note {
      margin: 16px 24px 0;
      padding: 12px 14px;
      border-radius: 10px;
      background: #fff7ed;
      border: 1px solid #fdba74;
      color: #9a3412;
    }
  </style>
</head>
<body>
  <div class="offline-note">
    Swagger UI assets are loaded from a CDN. If this page looks broken, open <code>/openapi.json</code> directly or use the practice UI at <code>/</code>.
  </div>
  <div id="swagger-ui"></div>
  <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
  <script>
    window.ui = SwaggerUIBundle({
      url: '/openapi.json',
      dom_id: '#swagger-ui',
      deepLinking: true,
      presets: [SwaggerUIBundle.presets.apis],
      layout: 'BaseLayout',
      tryItOutEnabled: true,
      displayRequestDuration: true,
      defaultModelsExpandDepth: 1
    });
  </script>
</body>
</html>"""


def render_practice_ui_html() -> str:
    """
    버튼을 눌러 API를 바로 테스트해볼 수 있는 작은 프론트엔드입니다.

    이 UI를 둔 이유:
    - Swagger가 없어도 브라우저에서 연동 감을 익힐 수 있습니다.
    - 프론트 코드에서 fetch()가 어떻게 쓰이는지 바로 볼 수 있습니다.
    - main.py를 건드리지 않고도 "백엔드 + 프론트 연결"을 실습할 수 있습니다.
    """
    return """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Practice API Lab</title>
  <style>
    :root {
      --bg: #f6f7fb;
      --panel: #ffffff;
      --line: #d7dce5;
      --text: #1f2937;
      --muted: #6b7280;
      --accent: #0f766e;
      --accent-soft: #ccfbf1;
      --danger: #b91c1c;
      --shadow: 0 18px 50px rgba(15, 23, 42, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", Arial, sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, #d1fae5 0, transparent 35%),
        radial-gradient(circle at top right, #e0f2fe 0, transparent 30%),
        var(--bg);
    }
    .wrap {
      max-width: 1120px;
      margin: 0 auto;
      padding: 36px 20px 48px;
    }
    h1 {
      margin: 0 0 10px;
      font-size: 34px;
      line-height: 1.15;
    }
    .lead {
      margin: 0 0 22px;
      color: var(--muted);
      font-size: 16px;
    }
    .links {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-bottom: 24px;
    }
    .links a {
      display: inline-block;
      padding: 10px 14px;
      border-radius: 999px;
      background: #ffffffcc;
      border: 1px solid var(--line);
      text-decoration: none;
      color: var(--text);
      box-shadow: var(--shadow);
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 16px;
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 18px;
      box-shadow: var(--shadow);
    }
    .method {
      display: inline-block;
      font-size: 12px;
      font-weight: 700;
      border-radius: 999px;
      padding: 6px 10px;
      margin-bottom: 10px;
    }
    .get { background: #dbeafe; color: #1d4ed8; }
    .post { background: #dcfce7; color: #166534; }
    h2 {
      margin: 0 0 8px;
      font-size: 20px;
    }
    p {
      margin: 0 0 12px;
      color: var(--muted);
      line-height: 1.5;
    }
    label {
      display: block;
      font-size: 13px;
      font-weight: 600;
      margin: 10px 0 6px;
    }
    input {
      width: 100%;
      padding: 10px 12px;
      border-radius: 10px;
      border: 1px solid var(--line);
      font: inherit;
      background: #fff;
    }
    button {
      margin-top: 12px;
      padding: 10px 14px;
      border: 0;
      border-radius: 10px;
      background: var(--accent);
      color: white;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
    }
    button:hover { filter: brightness(1.03); }
    pre {
      margin: 14px 0 0;
      padding: 12px;
      min-height: 120px;
      white-space: pre-wrap;
      word-break: break-word;
      border-radius: 12px;
      background: #0f172a;
      color: #e2e8f0;
      overflow: auto;
    }
    .hint {
      margin-top: 22px;
      padding: 16px 18px;
      background: var(--accent-soft);
      border: 1px solid #99f6e4;
      border-radius: 16px;
      line-height: 1.55;
    }
    .error { color: var(--danger); }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Practice API Lab</h1>
    <p class="lead">
      main.py와 별개로 돌리는 사이드 실습용 UI입니다. 버튼을 눌러 API 호출 결과를 확인하고,
      필요하면 Swagger에서도 같은 API를 테스트할 수 있습니다.
    </p>

    <div class="links">
      <a href="/docs" target="_blank" rel="noreferrer">Swagger 열기</a>
      <a href="/openapi.json" target="_blank" rel="noreferrer">OpenAPI JSON 보기</a>
    </div>

    <div class="grid">
      <section class="card">
        <span class="method get">GET</span>
        <h2>/api/status</h2>
        <p>서버 상태를 확인합니다.</p>
        <button onclick="callGet('/api/status', 'status-result')">호출하기</button>
        <pre id="status-result">아직 호출 전입니다.</pre>
      </section>

      <section class="card">
        <span class="method get">GET</span>
        <h2>/api/hello</h2>
        <p>가장 쉬운 GET 연습용 엔드포인트입니다.</p>
        <button onclick="callGet('/api/hello', 'hello-result')">호출하기</button>
        <pre id="hello-result">아직 호출 전입니다.</pre>
      </section>

      <section class="card">
        <span class="method get">GET</span>
        <h2>/api/document-count</h2>
        <p>샘플 문서 개수를 확인합니다.</p>
        <button onclick="callGet('/api/document-count', 'count-result')">호출하기</button>
        <pre id="count-result">아직 호출 전입니다.</pre>
      </section>

      <section class="card">
        <span class="method post">POST</span>
        <h2>/api/echo</h2>
        <p>이름을 보내면 그대로 응답합니다.</p>
        <label for="echo-name">name</label>
        <input id="echo-name" value="Yong">
        <button onclick="callPost('/api/echo', { name: byId('echo-name').value }, 'echo-result')">호출하기</button>
        <pre id="echo-result">아직 호출 전입니다.</pre>
      </section>

      <section class="card">
        <span class="method post">POST</span>
        <h2>/api/document-exists</h2>
        <p>샘플 문서 ID 존재 여부를 확인합니다.</p>
        <label for="document-id">document_id</label>
        <input id="document-id" value="doc-002">
        <button onclick="callPost('/api/document-exists', { document_id: byId('document-id').value }, 'exists-result')">호출하기</button>
        <pre id="exists-result">아직 호출 전입니다.</pre>
      </section>

      <section class="card">
        <span class="method post">POST</span>
        <h2>/api/add-numbers</h2>
        <p>숫자 두 개를 더합니다.</p>
        <label for="number-a">a</label>
        <input id="number-a" value="10">
        <label for="number-b">b</label>
        <input id="number-b" value="25">
        <button onclick="callPost('/api/add-numbers', { a: Number(byId('number-a').value), b: Number(byId('number-b').value) }, 'sum-result')">호출하기</button>
        <pre id="sum-result">아직 호출 전입니다.</pre>
      </section>
    </div>

    <div class="hint">
      Swagger는 있으면 편하지만 꼭 필수는 아닙니다.<br>
      프론트에서 <code>fetch()</code>로 정상 호출되고, 응답이 UI에 잘 보이면 이미 API 연동은 확인된 것입니다.<br>
      다만 Swagger는 요청 스펙, 예제 입력, 응답 구조를 정리해서 보는 데 유용해서 같이 두었습니다.
    </div>
  </div>

  <script>
    function byId(id) {
      return document.getElementById(id);
    }

    function printResult(id, data) {
      byId(id).textContent = JSON.stringify(data, null, 2);
    }

    async function callGet(url, outputId) {
      try {
        const response = await fetch(url, { cache: "no-store" });
        const payload = await response.json();
        printResult(outputId, { ok: response.ok, status: response.status, payload });
      } catch (error) {
        printResult(outputId, { ok: false, error: String(error) });
      }
    }

    async function callPost(url, body, outputId) {
      try {
        const response = await fetch(url, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        const payload = await response.json();
        printResult(outputId, { ok: response.ok, status: response.status, requestBody: body, payload });
      } catch (error) {
        printResult(outputId, { ok: false, error: String(error) });
      }
    }
  </script>
</body>
</html>"""


class PracticeApiHandler(BaseHTTPRequestHandler):
    """
    BaseHTTPRequestHandler를 상속해서 우리가 원하는 API 서버를 직접 만듭니다.

    이 프로젝트의 main.py도 큰 흐름은 비슷합니다.
    핵심 아이디어는 아래와 같습니다.

    - do_GET()  : GET 요청 처리
    - do_POST() : POST 요청 처리
    - 경로(path)를 보고 어떤 함수를 실행할지 라우팅
    - 각 _handle_* 함수에서 실제 비즈니스 로직 수행

    즉:
    "요청 분기"와 "실제 처리"를 나눠두는 구조입니다.

    이렇게 나누는 이유:
    - do_GET/do_POST는 깔끔하게 유지할 수 있습니다.
    - API가 늘어나도 _handle_* 함수 단위로 관리할 수 있습니다.
    - 테스트/디버깅할 때 어느 함수가 책임지는지 명확합니다.
    """

    # BaseHTTPRequestHandler는 기본 로그가 조금 장황합니다.
    # 학습용에서는 너무 시끄럽지 않게 최소한만 출력하도록 조정할 수 있습니다.
    # 지금은 기본 동작을 유지하되, 필요하면 log_message를 override하면 됩니다.

    def do_GET(self) -> None:
        """
        GET 요청이 들어오면 가장 먼저 실행되는 메서드입니다.

        설계 포인트:
        1. URL에서 path만 뽑는다.
        2. path별로 처리 함수를 연결한다.
        3. 모르는 경로는 404를 준다.
        4. 예상하지 못한 오류는 500으로 감싼다.
        """
        try:
            path = urlparse(self.path).path

            if path == "/":
                self._send_html(render_practice_ui_html(), status=200)
                return

            if path == "/docs":
                self._send_html(render_swagger_ui_html(), status=200)
                return

            if path == "/api/status":
                self._handle_status()
                return

            if path == "/api/hello":
                self._handle_hello()
                return

            if path == "/api/document-count":
                self._handle_document_count()
                return

            if path == "/openapi.json":
                self._send_json(build_openapi_like_spec(), status=200)
                return

            if path == "/openapi-practice.json":
                self._send_json(build_openapi_like_spec(), status=200)
                return

            self._send_json({"error": "not_found", "path": path}, status=404)
        except Exception as exc:
            self._send_json({"error": "internal_server_error", "detail": str(exc)}, status=500)

    def do_POST(self) -> None:
        """
        POST 요청이 들어오면 실행되는 메서드입니다.

        왜 GET/POST를 나누는가?
        - GET은 조회
        - POST는 데이터 전달, 생성, 계산, 실행

        이 프로젝트에서도 대체로 아래 규칙을 따릅니다.
        - GET  /api/documents
        - POST /api/query
        - POST /api/summarize
        - POST /api/run
        """
        try:
            path = urlparse(self.path).path

            if path == "/api/echo":
                self._handle_echo()
                return

            if path == "/api/document-exists":
                self._handle_document_exists()
                return

            if path == "/api/add-numbers":
                self._handle_add_numbers()
                return

            self._send_json({"error": "not_found", "path": path}, status=404)
        except Exception as exc:
            self._send_json({"error": "internal_server_error", "detail": str(exc)}, status=500)

    def _handle_status(self) -> None:
        """
        가장 단순한 상태 조회 API입니다.

        왜 상태 API가 필요한가?
        - 서버가 실행 중인지 빠르게 확인할 수 있습니다.
        - 프론트엔드가 초기 상태를 읽을 수 있습니다.
        - 운영 중 헬스체크 용도로도 자주 씁니다.

        GET /api/status
        응답 예시:
        {
            "ok": true,
            "service": "practice-api",
            "version": "1.0.0"
        }
        """
        self._send_json(
            {
                "ok": True,
                "service": "practice-api",
                "version": "1.0.0",
            },
            status=200,
        )

    def _handle_hello(self) -> None:
        """
        연습용 GET API입니다.

        이 API를 먼저 만들어보는 이유:
        - 입력값이 없어도 됩니다.
        - DB도 필요 없습니다.
        - 외부 API도 필요 없습니다.
        - "라우팅 -> 핸들러 -> JSON 응답"의 최소 구조를 익히기에 좋습니다.

        GET /api/hello
        응답 예시:
        {
            "message": "hello",
            "guide": "this is a practice GET API"
        }
        """
        self._send_json({

        }
        )

    def _handle_echo(self) -> None:
        """
        연습용 POST API입니다.

        POST API를 연습할 때 중요한 포인트:
        1. 요청 JSON을 읽는다.
        2. 필요한 필드가 있는지 검증한다.
        3. 없으면 400 에러를 반환한다.
        4. 있으면 응답 JSON을 만든다.

        요청 예시:
        {
            "name": "Yong"
        }

        성공 응답 예시:
        {
            "message": "안녕하세요, Yong",
            "received": {
                "name": "Yong"
            }
        }

        실패 응답 예시:
        {
            "error": "name_is_required"
        }

        왜 name 검증을 먼저 하는가?
        - 서버는 "필수 입력값이 없는 요청"을 가능한 빨리 거절해야 합니다.
        - 그래야 뒤에서 더 복잡한 로직이 잘못 실행되지 않습니다.
        - 현재 프로젝트의 query/summarize 핸들러들도 같은 패턴입니다.
        """
        payload = self._read_json_payload()
        name = str(payload.get("name", "")).strip()

        if not name:
            self._send_json({"error": "name_is_required"}, status=400)
            return

        self._send_json(
            {
                "message": f"안녕하세요, {name}",
                "received": {"name": name},
            },
            status=200,
        )

    def _handle_document_count(self) -> None:
        """
        직접 만들어보기 좋은 첫 번째 연습 API입니다.

        GET /api/document-count

        왜 이 API가 좋은 연습인가?
        - GET 요청 연습이 됩니다.
        - 입력값이 없어서 부담이 적습니다.
        - 응답 JSON 설계를 연습할 수 있습니다.
        - 나중에 main.py에서 self.manager.get_document_list()와 연결하는 감을 익히기 좋습니다.

        이 예제 파일은 실제 문서 저장소와 연결되어 있지 않기 때문에
        sample_documents라는 "가짜 데이터"를 사용합니다.
        실전에서는 보통 아래처럼 바뀝니다.

            documents = self.manager.get_document_list()
            count = len(documents)

        응답 예시:
        {
            "count": 3,
            "documents": ["doc-001", "doc-002", "doc-003"]
        }
        """
        sample_documents = self._get_sample_documents()
        self._send_json(
            {
            "count": 3,
            "documents": ["doc-001","doc-002", "doc-003"]
            }
            ,status=200
        )
        
    def _handle_document_exists(self) -> None:
        """
        직접 만들어보기 좋은 두 번째 연습 API입니다.

        POST /api/document-exists

        요청 예시:
        {
            "document_id": "doc-002"
        }

        응답 예시:
        {
            "document_id": "doc-002",
            "exists": true
        }

        이 API로 연습할 수 있는 것:
        - POST 요청 JSON 읽기
        - 필수값 검증
        - 조건 비교
        - boolean 응답 설계

        실전에서는 문서 목록을 DB나 파일 시스템, 또는 manager 객체에서 읽겠지만
        여기서는 sample_documents를 사용합니다.
        """
        payload = self._read_json_payload()
        document_id = str(payload.get("document_id", "")).strip()

        if not document_id:
            self._send_json({"error": "document_id_is_required"}, status=400)
            return

        sample_documents = self._get_sample_documents()
        exists = document_id in sample_documents

        self._send_json(
            {
                "document_id": document_id,
                "exists": exists,
            },
            status=200,
        )

    def _handle_add_numbers(self) -> None:
        """
        직접 만들어보기 좋은 세 번째 연습 API입니다.

        POST /api/add-numbers

        요청 예시:
        {
            "a": 10,
            "b": 25
        }

        응답 예시:
        {
            "a": 10.0,
            "b": 25.0,
            "sum": 35.0
        }

        이 API가 좋은 이유:
        - 문자열 검증이 아니라 숫자 검증을 연습할 수 있습니다.
        - 형 변환(float) 처리 연습이 됩니다.
        - 계산 결과를 JSON으로 반환하는 패턴을 익힐 수 있습니다.
        - 나중에 더 복잡한 "요청 -> 계산 -> 응답" API로 확장하기 쉽습니다.
        """
        payload = self._read_json_payload()

        if "a" not in payload or "b" not in payload:
            self._send_json({"error": "a_and_b_are_required"}, status=400)
            return

        try:
            a = float(payload["a"])
            b = float(payload["b"])
        except (TypeError, ValueError):
            self._send_json({"error": "a_and_b_must_be_numbers"}, status=400)
            return

        self._send_json(
            {
                "a": a,
                "b": b,
                "sum": a + b,
            },
            status=200,
        )

    def _get_sample_documents(self) -> list[str]:
        """
        실습용 가짜 문서 목록입니다.

        왜 이런 샘플 함수를 두는가?
        - 지금 파일은 실제 프로젝트의 manager나 DB에 연결하지 않았습니다.
        - 그래도 "문서 목록을 조회해서 처리한다"는 감각은 유지하고 싶었습니다.
        - 그래서 샘플 데이터를 한 곳에 모아두고 재사용하게 만들었습니다.

        실전에서는 이 함수 대신 아래 같은 실제 코드가 들어갈 가능성이 큽니다.

            return [item["document_id"] for item in self.manager.get_document_list()]
        """
        return ["doc-001", "doc-002", "doc-003"]

    def _read_json_payload(self) -> dict[str, Any]:
        """
        POST 요청 본문(body)에서 JSON을 읽는 공통 함수입니다.

        왜 공통 함수로 분리하는가?
        - 여러 API에서 같은 코드가 반복됩니다.
        - JSON 읽기 실패 처리를 한 곳에서 통일할 수 있습니다.
        - 핸들러 함수(_handle_echo 등)가 훨씬 읽기 쉬워집니다.

        동작 방식:
        1. Content-Length 헤더를 읽어서 바디 길이를 확인
        2. wfile가 아니라 rfile에서 raw bytes를 읽음
        3. utf-8 문자열로 디코드
        4. json.loads()로 dict로 변환

        에러 처리:
        - 비어 있으면 400
        - JSON 형식이 아니면 400
        - object(dict)가 아니면 400
        """
        content_length = int(self.headers.get("Content-Length", "0") or "0")
        if content_length <= 0:
            raise ValueError("request_body_is_required")

        raw = self.rfile.read(content_length)
        text = raw.decode("utf-8")

        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid_json: {exc.msg}") from exc

        if not isinstance(payload, dict):
            raise ValueError("json_object_body_is_required")

        return payload

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        """
        JSON 응답을 보내는 공통 함수입니다.

        왜 이것도 공통 함수로 두는가?
        - Content-Type 헤더를 매번 쓰지 않아도 됩니다.
        - UTF-8 인코딩을 매번 중복 작성하지 않아도 됩니다.
        - 응답 형식을 프로젝트 전체에서 일관되게 맞출 수 있습니다.
        """
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


    def _send_html(self, html: str, status: int = 200) -> None:
        """
        HTML 응답을 보내는 공통 함수입니다.

        이번 실습에서는 API만 보는 것이 아니라
        브라우저에서 버튼을 눌러 API를 호출하는 작은 UI도 함께 제공하기 때문에
        JSON 전송 함수와 별도로 HTML 전송 함수를 둡니다.
        """
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    

def call_openai_responses_example(prompt: str) -> dict[str, Any]:
    """
    외부 API 호출 예시 함수입니다.

    이 함수는 현재 프로젝트의 openai_answerer.py와 비슷한 모양을
    학습용으로 단순화한 버전입니다.

    이 함수를 별도로 둔 이유:
    - "우리 서버 내부 API"와
      "우리 서버가 외부 서비스를 호출하는 코드"는 역할이 다르기 때문입니다.

    역할 구분:
    - /api/echo 같은 것은 우리가 제공하는 내부 API
    - call_openai_responses_example() 같은 것은 외부 OpenAI API 호출 함수

    설계 순서:
    1. 환경변수에서 API 키를 읽는다.
    2. 요청 payload를 만든다.
    3. Authorization 헤더를 넣는다.
    4. POST 요청을 보낸다.
    5. JSON 응답을 파싱한다.

    주의:
    - 이 함수는 예제입니다.
    - 이 파일 어디에서도 자동 호출하지 않습니다.
    - 실제 사용 시에는 예외 처리, 재시도, 타임아웃 전략을 더 정교하게 잡아야 합니다.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")

    payload = {
        "model": os.environ.get("OPENAI_MODEL", "gpt-5.2"),
        "input": prompt,
        "reasoning": {"effort": "low"},
        "text": {"verbosity": "low"},
    }

    req = urllib_request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib_request.urlopen(req, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib_error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API request failed: {exc.code} {detail}") from exc


def run_server() -> None:
    """
    서버 시작 함수입니다.

    ThreadingHTTPServer를 쓰는 이유:
    - 요청을 병렬로 처리하기 쉽습니다.
    - 학습용으로도 구조가 단순합니다.
    - 현재 프로젝트 스타일과도 잘 맞습니다.
    """
    server = ThreadingHTTPServer((HOST, PORT), PracticeApiHandler)
    print(f"Practice API server started: http://{HOST}:{PORT}")
    print(f"Practice UI:                http://{HOST}:{PORT}/")
    print(f"Swagger UI:                 http://{HOST}:{PORT}/docs")
    print(f"OpenAPI JSON:               http://{HOST}:{PORT}/openapi.json")
    print(f"GET practice endpoint:      http://{HOST}:{PORT}/api/hello")
    print(f"Status endpoint:            http://{HOST}:{PORT}/api/status")
    print("POST practice endpoints:    /api/echo, /api/document-exists, /api/add-numbers")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    run_server()
