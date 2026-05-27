# Token Flow Exploration: Articraft RL Environment

**Date**: 2026-05-26  
**Task**: Understand how tokens flow through articraft RL environment to plan context window management

## Key Findings

### 1. File Locations & Function Signatures

| File | Location | Key Functions |
|------|----------|----------------|
| **prompts.py** | `prime-rl/environments/articraft/articraft_env/prompts.py` | `load_system_prompt()`, `build_turn0_messages()` |
| **env.py** | `prime-rl/environments/articraft/articraft_env/env.py` | `setup_state()`, `env_response()` |
| **workspace_docs.py** | `articraft/agent/workspace_docs.py` | `load_sdk_docs_reference()`, `load_sdk_docs_bundle()` |
| **compaction_policy.py** | `articraft/agent/providers/compaction_policy.py` | `decide_compaction()`, `pressure_ratio()` |
| **blendergym/env.py** | `prime-rl/environments/blendergym/blendergym/env.py` | `setup_state()`, `get_prompt_messages()` |

### 2. Token Sizes

#### SDK Documentation
- **Preloaded (Turn 0)**: 34.5 KB, ~956 lines
  - quickstart.md: 9.5 KB
  - probe-tooling.md: 8.6 KB
  - testing.md: 16.3 KB
- **Full docs (via read_file)**: 224 KB, 7,255 lines
  - 23 doc files mapped in `_DOC_PATH_ALIASES`
  - Only preloaded docs in Turn 0 to save tokens

#### Message Accumulation (Worst Case)
- **Turn 0**: ~2,300-4,500 tokens (system + preload + task)
- **Per turn**: ~150-700 tokens (tool calls + results)
- **50 turns max**: ~39,500 tokens worst case

### 3. Message Flow Architecture

#### Turn 0 (`setup_state`)
```
state["prompt"] = [
    *system_msgs,           # From verifiers (frozen)
    *build_turn0_messages(
        task.prompt_text,
        sdk_docs_context=self.sdk_docs_context,
        provider=self.provider,
    )
]
```

Structure:
- Message 1 (role=user): SDK docs context (preloaded)
- Message 2 (role=user): Runtime guidance + task prompt

#### Per Turn (`env_response`)
```python
result_messages: list[dict[str, Any]] = []
for tc in tool_calls:
    result_messages.append({
        "role": "tool",
        "content": json.dumps(result.to_dict()),
        "tool_call_id": tc_id,
    })
```

**NO TRUNCATION**: Messages accumulate linearly per turn

### 4. Existing Truncation/Compaction

#### ArticraftEnv (env.py)
- ✅ Compile result freshness cache (avoids redundant output)
- ❌ NO token counting
- ❌ NO explicit truncation
- ❌ NO compaction hooks

#### CompactionPolicy (articraft codebase)
- **Pressure bands**: 55%, 70%, 85% thresholds
- **Hard valve**: 90% of hard_threshold
- **Decision logic**: Compile plateau + consecutive failures
- **Cooldown**: 2 turns between compactions
- **Usage**: Orchestrator layer (not env.py)

#### BlenderGym Comparison
- Max turns: **3** (implicit limit)
- Message rebuild from trajectory (not streaming)
- Image encoding overhead: ~500-1000 tokens per image
- No explicit token counting

### 5. Token Management Gaps

**What's missing in ArticraftEnv**:
1. Token counter at env init
2. Per-turn token budget tracking
3. Soft pruning strategy
4. Compaction policy integration
5. Anthropic prompt caching annotations
6. Configuration for context limits

### 6. Pressure Ratio Decision Table

| Trigger Band | Min Pressure | Min Failures | Min Items | Use Case |
|---|---|---|---|---|
| `high_pressure` | 85% | 3 | 2 | About to hit ceiling |
| `medium_pressure` | 70% | 4 | 2 | High usage |
| `early_pressure` | 55% | 5 | 3 | Preventive |
| `hard_pressure` | 90% | N/A | N/A | Safety valve |

**Cache boost**: If 60%+ cached, require +1 failure streak

---

## Next Steps for Context Window Management

1. **Add token tracking to ArticraftEnv**
   - Counter at `__init__`: system + SDK docs
   - Per-turn counters in `env_response()`
   
2. **Integrate compaction policy**
   - Check `decide_compaction()` in `env_response()`
   - Wire pressure_ratio tracking
   
3. **Config support**
   - `[context] max_prompt_tokens = 180000`
   - `enable_compaction = true`
   - `track_token_usage = true`

4. **Cache optimization**
   - Use Anthropic prompt caching for system + preloaded docs
   - Eligible for cache: Static content across rollouts
