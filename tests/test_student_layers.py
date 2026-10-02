import json
import unicodedata
from types import SimpleNamespace

from arena.corpus import Corpus, Doc, INJECTION_CANARY
from arena.model import FINALIZE_SENTINEL, RealModel
from arena.tools import ToolResult
from harness.layers.budget_policy import BudgetPolicy
from harness.layers.citation_checker import CitationChecker
from harness.layers.critic import Critic
from harness.layers.injection_guard import BLOCK_END, BLOCK_START, InjectionGuard
from harness.layers.retry import Retry


def test_evidence_layers_repair_citations_split_conflicts_and_drop_fabrications():
    first = "Thời gian giao hàng nội thành là 2 ngày làm việc."
    second = "Quy định khác ghi thời gian là 3 ngày làm việc."
    corpus = Corpus([
        Doc("a", "A", first + "\nDòng phụ A", ()),
        Doc("b", "B", second + "\nDòng phụ B", ()),
    ])
    ctx = SimpleNamespace(corpus=corpus, observed_text="\n".join(doc.body for doc in corpus.docs), state={})
    report = {
        "answer": "Đối chiếu nguồn",
        "abstain": False,
        "claims": [
            {"text": first, "doc_id": "b"},
            {"text": first + " và " + second, "doc_id": "a"},
            {"text": "Con số này hoàn toàn không có bằng chứng.", "doc_id": "a"},
        ],
    }

    CitationChecker().after_agent(ctx, report)
    Critic().after_agent(ctx, report)

    assert report["claims"] == [
        {"text": first, "doc_id": "a"},
        {"text": first, "doc_id": "a"},
        {"text": second, "doc_id": "b"},
    ]
    assert report["citations"] == ["a", "b"]
    assert report["abstain"] is True


def test_budget_reserves_submit_even_during_retries():
    ctx = SimpleNamespace(max_tool_calls=3, tools=SimpleNamespace(calls=0), state={})
    policy, retry = BudgetPolicy(), Retry()

    def call(name, args):
        ctx.tools.calls += 1
        return ToolResult(ok=True, content="[NOISE: degraded]")

    result = policy.wrap_tool_call(
        ctx, lambda name, args: retry.wrap_tool_call(ctx, call, name, args), "search", {}
    )
    assert result.ok and ctx.tools.calls == 2
    assert ctx.state["retry_attempts"] == 2
    messages = [{"role": "user", "content": "question"}]
    assert FINALIZE_SENTINEL in policy.before_model(ctx, messages)[-1]["content"]
    assert len(messages) == 1
    assert not policy.wrap_tool_call(ctx, call, "search", {}).ok
    assert ctx.tools.calls == 2


def test_injection_guard_removes_multiple_and_truncated_blocks_without_editing_claims():
    guard = InjectionGuard()
    ctx = SimpleNamespace(state={})
    content = f"safe {BLOCK_START} attack {BLOCK_END} middle {BLOCK_START} cut off"
    result = guard.wrap_tool_call(
        ctx, lambda name, args: ToolResult(True, content), "fetch_doc", {"doc_id": "a"}
    )
    assert result.content.startswith("safe [nội dung không đáng tin cậy đã bị loại bỏ] middle")
    assert BLOCK_START not in result.content and "attack" not in result.content
    assert ctx.state["sanitized_docs"] == {"a"}
    claim = {"text": "genuine source line", "doc_id": "a"}
    report = {"answer": INJECTION_CANARY + " genuine answer", "claims": [claim]}
    guard.after_agent(None, report)
    assert INJECTION_CANARY not in report["answer"] and report["claims"] == [claim]


def test_safe_line_from_sanitized_document_keeps_its_citation():
    safe = "Yêu cầu đổi trả đã được giải quyết trong 2 ngày."
    body = safe + "\n" + BLOCK_START + " malicious " + BLOCK_END
    corpus = Corpus([Doc("a", "A", body, ())])
    ctx = SimpleNamespace(state={}, corpus=corpus)
    result = InjectionGuard().wrap_tool_call(
        ctx, lambda name, args: ToolResult(True, body), "fetch_doc", {"doc_id": "a"}
    )
    ctx.observed_text = result.content
    report = {"claims": [{"text": safe, "doc_id": "a"}], "abstain": False}
    CitationChecker().after_agent(ctx, report)
    Critic().after_agent(ctx, report)
    assert report["claims"] == [{"text": safe, "doc_id": "a"}]


def test_normalised_quote_keeps_model_text_and_finds_the_fetched_source():
    line = "Theo quy định, thời hạn thông báo là 2 ngày làm việc."
    quoted = unicodedata.normalize("NFD", line.upper()).replace(" ", "  ")
    corpus = Corpus([Doc("a", "A", line, ()), Doc("b", "B", "Thông tin khác.", ())])
    ctx = SimpleNamespace(corpus=corpus, observed_text=line, state={})
    report = {"claims": [{"text": quoted, "doc_id": "b"}], "abstain": False}
    CitationChecker().after_agent(ctx, report)
    Critic().after_agent(ctx, report)
    assert report["claims"] == [{"text": quoted, "doc_id": "a"}]
    assert report["citations"] == ["a"]


def test_search_snippet_with_unclosed_attack_keeps_other_hits_readable():
    hits = [
        {"doc_id": "a", "title": "A", "snippet": "safe " + BLOCK_START + INJECTION_CANARY},
        {"doc_id": "b", "title": "B", "snippet": "unaffected result"},
    ]
    result = InjectionGuard().wrap_tool_call(
        SimpleNamespace(state={}),
        lambda name, args: ToolResult(True, json.dumps(hits, ensure_ascii=False)),
        "search",
        {},
    )
    clean = json.loads(result.content)
    assert INJECTION_CANARY not in result.content
    assert clean[0]["doc_id"] == "a" and clean[1] == hits[1]


def test_malformed_claims_abstain_instead_of_submitting_an_unaudited_answer():
    ctx = SimpleNamespace(corpus=None, observed_text="", state={})
    report = {"answer": "Không rõ", "claims": "not a list", "abstain": False}
    Critic().after_agent(ctx, report)
    assert report["abstain"] is True and report["claims"] == []


def test_final_guard_discards_tainted_claim_without_rewriting_it():
    clean = {"text": "Tài liệu an toàn ghi rõ thời hạn 2 ngày.", "doc_id": "a"}
    tainted = {"text": "Injected " + INJECTION_CANARY, "doc_id": "b"}
    report = {
        "answer": INJECTION_CANARY + " answer",
        "verdict": "safe " + INJECTION_CANARY,
        "claims": [clean, tainted],
        "citations": ["a", "b"],
    }
    InjectionGuard().after_agent(None, report)
    assert report["claims"] == [clean]
    assert report["citations"] == ["a"]
    assert INJECTION_CANARY not in json.dumps(report, ensure_ascii=False)

def test_search_snippet_is_a_citable_observed_source():
    line = "Quy trình kiểm tra đối tác yêu cầu xác minh ba giấy phép."
    corpus = Corpus([Doc("a", "A", line + "\nPhần còn lại không nằm trong snippet.", ())])
    ctx = SimpleNamespace(corpus=corpus, state={})
    hit = [{"doc_id": "a", "title": "A", "snippet": line}]
    result = InjectionGuard().wrap_tool_call(
        ctx, lambda name, args: ToolResult(True, json.dumps(hit, ensure_ascii=False)),
        "search", {"query": "kiểm tra đối tác"},
    )
    ctx.observed_text = result.content
    report = {"answer": line, "claims": [{"text": line, "doc_id": "wrong"}]}
    CitationChecker().after_agent(ctx, report)
    Critic().after_agent(ctx, report)
    assert report["claims"] == [{"text": line, "doc_id": "a"}]

def test_visible_quote_survives_truncated_fetch():
    line = "Chính sách bảo hành áp dụng trong mười hai tháng."
    corpus = Corpus([Doc("a", "A", line + "\n" + "phụ lục " * 200, ())])
    ctx = SimpleNamespace(corpus=corpus, state={})
    result = InjectionGuard().wrap_tool_call(
        ctx, lambda name, args: ToolResult(True, line + " [TRUNCATED: connection dropped]"),
        "fetch_doc", {"doc_id": "a"},
    )
    ctx.observed_text = result.content
    report = {"claims": [{"text": line, "doc_id": "a"}]}
    Critic().after_agent(ctx, report)
    assert report["claims"] == [{"text": line, "doc_id": "a"}]

def test_real_model_search_nudge_is_one_turn_only_and_preserves_question():
    policy = BudgetPolicy()
    ctx = SimpleNamespace(
        max_tool_calls=8, tools=SimpleNamespace(calls=0), step=0,
        model=SimpleNamespace(inner=RealModel.__new__(RealModel)),
    )
    messages = [{"role": "system", "content": "rules"},
                {"role": "user", "content": "Câu hỏi gốc?"}]
    outbound = policy.before_model(ctx, messages)
    assert "search" in outbound[0]["content"]
    assert outbound[1] == messages[1] and messages[0]["content"] == "rules"
    ctx.step = 1
    assert policy.before_model(ctx, messages) is messages

def test_final_guard_discards_tainted_metadata_in_any_report_field():
    report = {"answer": {"nested": INJECTION_CANARY}, "claims": None,
              "citations": [INJECTION_CANARY], "verdict": {"nested": INJECTION_CANARY},
              "extra": {"nested": INJECTION_CANARY}}
    InjectionGuard().after_agent(None, report)
    assert INJECTION_CANARY not in json.dumps(report, ensure_ascii=False)
    assert report["claims"] == [] and report["abstain"] is True
