# -*- coding: utf-8 -*-
"""OpenAI API key 연결 테스트 스크립트.
실행: python test_openai_key.py
"""
import json
import os
import sys
from pathlib import Path
from urllib import error, request


def load_env() -> dict:
    values = {}
    for name in (".env.local", ".env"):
        path = Path(__file__).parent / name
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            values[k.strip()] = v.strip().strip('"').strip("'")
    return values


def test_connection(api_key: str, model: str) -> None:
    print(f"  API Key : {api_key[:12]}...{api_key[-4:]}")
    print(f"  Model   : {model}")
    print()

    # 1) embeddings 엔드포인트 테스트
    print("[1] Embeddings API 테스트 (text-embedding-3-small) ...")
    emb_url = "https://api.openai.com/v1/embeddings"
    emb_payload = json.dumps({"model": "text-embedding-3-small", "input": ["테스트"]}).encode()
    emb_req = request.Request(
        emb_url,
        data=emb_payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(emb_req, timeout=15) as resp:
            body = json.loads(resp.read())
        vec_len = len(body["data"][0]["embedding"])
        print(f"  OK - 임베딩 벡터 차원: {vec_len}")
    except error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        print(f"  FAIL (HTTP {e.code}) — {detail[:300]}")
    except error.URLError as e:
        print(f"  FAIL (URLError) — {e.reason}")
        print("  → 네트워크/프록시 문제일 수 있습니다.")
    except Exception as e:
        print(f"  FAIL - {e}")

    print()

    # 2) chat completions 엔드포인트 테스트
    print(f"[2] Chat Completions API 테스트 (model={model}) ...")
    chat_url = "https://api.openai.com/v1/chat/completions"
    chat_payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "안녕"}],
        "max_completion_tokens": 5,
    }).encode()
    chat_req = request.Request(
        chat_url,
        data=chat_payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(chat_req, timeout=20) as resp:
            body = json.loads(resp.read())
        reply = body["choices"][0]["message"]["content"]
        print(f"  OK - 응답: {reply!r}")
    except error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        print(f"  FAIL (HTTP {e.code}) — {detail[:300]}")
        if e.code == 401:
            print("  → API 키가 잘못됐거나 만료되었습니다.")
        elif e.code == 404:
            print(f"  → 모델 '{model}'이 존재하지 않습니다. .env의 OPENAI_MODEL을 확인하세요.")
        elif e.code == 429:
            print("  → 요청 한도(rate limit) 초과 또는 크레딧 부족입니다.")
    except error.URLError as e:
        print(f"  FAIL (URLError) — {e.reason}")
        print("  → 네트워크/프록시 문제일 수 있습니다.")
    except Exception as e:
        print(f"  FAIL - {e}")


def main() -> None:
    env = load_env()
    api_key = os.environ.get("OPENAI_API_KEY") or env.get("OPENAI_API_KEY", "")
    model = os.environ.get("OPENAI_MODEL") or env.get("OPENAI_MODEL", "gpt-4o")

    if not api_key:
        print("ERROR: OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
        sys.exit(1)

    print("=== OpenAI API Key 연결 테스트 ===")
    test_connection(api_key, model)
    print()
    print("완료.")


if __name__ == "__main__":
    main()
