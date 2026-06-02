# Inference 日志噪声排查参考

记录长跑 RL 训练里看起来很吓人但**功能正常**的 inference.log 错误模式，避免误判为故障。

---

## `/v1/tokenize` 400 "No user query found in messages"（35% 请求噪声）

### 何时会看到

- 任何用 verifiers + Qwen3.5 (chat_template) 跑 RL 训练的 run
- inference.log 中以 `ERROR ... POST /v1/tokenize HTTP/1.1 400 Bad Request` 形式出现，伴随完整 Python traceback
- 11h 的 articraft v2 训练中累积 **65,430 次 400 / 196,290 行 traceback**，占总请求数 **~35%**

### 是否需要修

**不影响训练正确性、不导致 crash、不占内存**。只是噪声大、让 inference.log 体积虚胖（798MB / 11h，其中 ~700MB 是这些 traceback）。

### 根因

verifiers 客户端的 **TITO 优化路径** (`openai_chat_completions_token_client.py`) 为了拿 base ids 桥接，会调一次：

```python
bridge_base_ids = await self.tokenize(messages=[dummy_assistant], ...)
```

这条 messages 里**只有 assistant、没有 user**。Qwen3.5 chat_template (tokenizer_config.json L66-78) 显式校验：

```jinja
{% if not has_user_query %}
  {{ raise_exception("No user query found in messages.") }}
{% endif %}
```

→ vLLM `/tokenize` 端点抛 `ValueError` → FastAPI 中间件转 HTTP 400 → 客户端的 `try: ...; except Exception: return None` 静默吞掉，fallback 到 MITO 路径（`/v1/chat/completions`，正常 200 OK 返回回复）。

实际生成全部走 MITO，**TITO 优化等于失效**。

### 排查时怎么甄别它不是真问题

| 检查 | 噪声特征 |
|---|---|
| `grep -c "POST /v1/chat/completions.*500" inference.log` | **0**（这才是真问题，v1 训练 11h crash 的就是这个）|
| `grep -c "POST /tokenize.*400" inference.log` | 几万级（噪声）|
| 错误率随时间变化 | 完全均匀，无 spike，无 burst |
| 单秒最大错误数 | 训练 11h 最高峰 67/sec，远低于 vLLM ApiServer 队列容量 |
| ApiServer 进程是否存活 | 6 个 PID 全程稳定，0 次 respawn |

### 与 v1 训练 11h crash 的关系

**注意区别**：v1 训练（articraft-0527-phase2-compaction）11h 后死掉，根因是同样的 `No user query found` 错误**走 `/v1/chat/completions` 端点返回 500 ISE**，触发 vLLM ApiServer 进程级 crash 然后整个 job SIGTERM 级联。

v2 vLLM 升级后**这条路径已经修了**：同样的输入现在走 `/tokenize` 返回 400（HTTP 错误，但 ApiServer 不死），客户端 except 兜住，训练继续。所以看到 v2 的 400 噪声**不要恐慌**，它是 v1 致命路径的"缓和版本"。

详细分析见 `.agents/session/2026-06-02-*-upstream-sync*.md`（如果存在）。

### 修复方案（独立于训练，优先级 P1）

如果想彻底消除噪声，最小修改是在 verifiers 客户端**短路**这次注定失败的请求：

```python
# verifiers/clients/openai_chat_completions_token_client.py
async def _tokenize_bridge(self, messages):
    # dummy_assistant 单条消息永远会触发 chat template 的 "No user query" 校验
    # 直接 return None，跳过这次注定失败的请求
    if not any(m["role"] == "user" for m in messages):
        return None
    return await self.tokenize(messages=messages, ...)
```

效果：
- 65k 个 400 噪声归零，inference.log 体积砍一半
- TITO fallback 行为不变（本来就是 None）
- 下次训练 crash 时日志能更快定位真问题

也可以走 `0b8c35e9` 上游引入的 **per-model client-side renderers + `/v1/generate`** 完全旁路这条路径。但启用前必须做 token-id parity test 验证 renderer chat template 与 vLLM 服务端 `qwen3_coder` parser 输出字节一致。

---

## 模板 `cache_position` `[ERROR]` 刷屏

orchestrator.log / trainer.log 启动后都会刷大量：

```
[ERROR] `cache_position` is part of Glm4MoeForCausalLM.forward's signature, but not documented...
[ERROR] `cache_position` is part of LlamaForCausalLM.forward's signature, but not documented...
```

来自 HuggingFace transformers 的 `auto_docstring` warning，针对每个注册的 model class 重复检查。**完全无害**。orchestrator.log 中 99%+ 的行可能都是这条。grep 时记得 `-v "cache_position"` 过滤。

上游已在更新版 transformers 中 silenced（fix 见 `0fca0d15 fix(models): document cache_position to silence auto_docstring warnings`），但需要相应的 transformers rev bump。
