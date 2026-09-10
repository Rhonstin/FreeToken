"""MTP speculative-decode driver (419.7): depth selection, draft placement, verify wiring.

Every CPU-runnable seam of the driver: the depth/clamp math (plain decode stays
inert), the adaptive per-request policy, the token-pool placement map, the
stub-model draft loop, and the verify -> finalize -> accounting chain against a real
(radix, CPU) CacheManager. GPU-only paths (draft forward, CUDA graphs) are gated and
marked, never faked.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from freetoken.core import Batch, Req, SamplingParams
from freetoken.distributed import DistributedInfo
from freetoken.engine.config import EngineConfig, apply_mtp_override
from freetoken.engine.mtp import (
    AdaptiveDepthPolicy,
    MtpConfig,
    clamp_spec_depth,
    collect_mtp_expert_pieces,
    draft_forward,
    draft_write_positions,
    make_spec_depth_fn,
    mtp_bank_layer_ids,
    propose_drafts,
    qsa_ring_capacity_for_depth,
    verify_and_finalize,
    write_draft_tokens,
)
from freetoken.engine.spec import SpecAccounting
from freetoken.scheduler.cache import CacheManager
from freetoken.scheduler.status import SchedulerStatusReporter, _spec_msg
from freetoken.scheduler.table import TableManager

V = 6


def _params(**overrides) -> SamplingParams:
    base = dict(temperature=0.0, top_k=-1, top_p=1.0, max_tokens=64)
    base.update(overrides)
    return SamplingParams(**base)


def _req(uid=1, prompt=16, output=16, cached=8, depth=0) -> Req:
    req = Req(
        input_ids=torch.arange(prompt, dtype=torch.int64),
        table_idx=uid,
        cached_len=cached,
        output_len=output,
        uid=uid,
        sampling_params=_params(),
        cache_handle=None,
    )
    req.cached_len = cached
    req.device_len = cached + 1
    req.spec_depth = 0
    if depth:
        req.reserve_spec(depth)
    return req


def _onehot(token: int, scale: float = 10.0) -> torch.Tensor:
    return F.one_hot(torch.tensor(token), V).float() * scale


def _managers(page_size=4, num_pages=32, num_reqs=4):
    table_manager = TableManager(
        num_reqs, torch.zeros((num_reqs, num_pages * page_size), dtype=torch.int32))
    cm = CacheManager(num_pages, page_size, table_manager.page_table, "radix")
    return table_manager, cm


# ======================================================================================
# MtpConfig + clamp_spec_depth: plain decode stays inert
# ======================================================================================


def test_mtp_config_defaults_off():
    cfg = MtpConfig()
    assert (cfg.depth, cfg.adaptive, cfg.enabled) == (0, False, False)
    assert MtpConfig(depth=3).enabled


def test_mtp_config_from_config_object():
    assert MtpConfig.from_config(SimpleNamespace(mtp_depth=2, mtp_adaptive=True)) == (
        MtpConfig(depth=2, adaptive=True))
    assert MtpConfig.from_config(SimpleNamespace()) == MtpConfig()
    assert MtpConfig.from_config(SimpleNamespace(mtp_depth=None)) == MtpConfig()


def test_clamp_spec_depth():
    assert clamp_spec_depth(3, cached_len=10, max_device_len=32) == 3
    # output budget: cached 10 + 1 anchor + 3 drafts commits at most index 14 = max - 1;
    # one slot past the frame stays free (a deeper commit would index past the id buffer)
    assert clamp_spec_depth(8, cached_len=10, max_device_len=14) == 2
    # no room past the anchor clamps to a plain row, never negative
    assert clamp_spec_depth(4, cached_len=13, max_device_len=14) == 0
    assert clamp_spec_depth(-2, cached_len=10, max_device_len=32) == 0


# ======================================================================================
# make_spec_depth_fn: None (not a zero-fn) keeps the manager's plain path
# ======================================================================================


def test_make_spec_depth_fn_none_when_off():
    assert make_spec_depth_fn(0) is None
    assert make_spec_depth_fn(-1) is None
    assert make_spec_depth_fn(0, adaptive=True) is None


def test_make_spec_depth_fn_clamps_each_request_to_its_budget():
    fn = make_spec_depth_fn(4)
    assert fn.policy is None
    assert fn(_req(cached=8, depth=0)) == 4  # room: 32 - 8 - 2
    tight = _req(prompt=14, output=1, cached=13)  # max 15: room for 0 drafts (one slot free)
    assert fn(tight) == 0
    # last output token: the frame is re-established from cached_len post-commit
    tight.cached_len = 14
    tight.device_len = 15
    tight.spec_depth = 0
    assert fn(tight) == 0


# ======================================================================================
# AdaptiveDepthPolicy: per-request depth tracks acceptance
# ======================================================================================


def test_adaptive_policy_starts_at_max_and_holds_partial():
    policy = AdaptiveDepthPolicy(3)
    assert policy.depth_for(7) == 3
    assert policy.record(7, n_accepted=1, depth=2) == 3
    assert policy.depth_for(7) == 3


def test_adaptive_policy_grows_on_full_accept_capped_at_max():
    policy = AdaptiveDepthPolicy(2)
    policy.record(1, n_accepted=1, depth=2)  # partial: hold at 2
    assert policy.depth_for(1) == 2
    assert policy.record(1, n_accepted=2, depth=2) == 2  # full: grow, capped at max
    roomy = AdaptiveDepthPolicy(4)
    roomy.record(1, n_accepted=0, depth=4)  # 4 -> 3
    assert roomy.record(1, n_accepted=3, depth=3) == 4  # full: grow back


def test_adaptive_policy_shrinks_on_all_reject_floored_at_zero():
    policy = AdaptiveDepthPolicy(3)
    assert policy.record(1, n_accepted=0, depth=3) == 2
    assert policy.record(1, n_accepted=0, depth=2) == 1
    assert policy.record(1, n_accepted=0, depth=1) == 0
    assert policy.record(1, n_accepted=0, depth=1) == 0  # floor, never negative


def test_adaptive_policy_ignores_zero_depth_steps():
    policy = AdaptiveDepthPolicy(3)
    policy.record(1, n_accepted=0, depth=0)  # clamped-to-plain: no signal
    assert policy.depth_for(1) == 3


def test_make_spec_depth_fn_adaptive_uses_the_policy():
    policy = AdaptiveDepthPolicy(3)
    fn = make_spec_depth_fn(3, adaptive=True, policy=policy)
    req = _req(uid=9, cached=8)
    assert fn(req) == 3
    policy.record(9, n_accepted=0, depth=3)
    assert fn(req) == 2


def test_make_spec_depth_fn_builds_a_policy_when_adaptive():
    fn = make_spec_depth_fn(2, adaptive=True)
    assert isinstance(fn.policy, AdaptiveDepthPolicy)
    assert fn.policy.max_depth == 2


# ======================================================================================
# Draft placement: positions + token-pool writes the verify forward reads
# ======================================================================================


def test_draft_write_positions_follow_the_anchor():
    assert draft_write_positions(10, 3) == [11, 12, 13]
    assert draft_write_positions(10, 0) == []


def test_write_draft_tokens_lands_where_make_write_tuple_points():
    from freetoken.scheduler.scheduler import _make_write_tuple

    pool = torch.zeros(4, 32, dtype=torch.int32)
    req = _req(uid=1, cached=10, depth=2)
    batch = Batch(reqs=[req], phase="decode")
    batch.padded_reqs = batch.reqs
    # a scheduled spec batch carries the anchor + draft rows flat in spec_rows order
    batch.spec_rows = [(req, 0), (req, 1), (req, 2)]
    _, write_pos = _make_write_tuple(batch, torch.device("cpu"))
    # the drafts are the inputs at [cached+1, cached+1+depth); the write map holds
    # each row's continuation one past it -- placement must precede the frame end
    assert write_pos.tolist() == [11, 12, 13]
    drafts = torch.tensor([21, 22], dtype=torch.int32)
    write_draft_tokens(pool, req.table_idx, draft_write_positions(10, 2), drafts)
    assert pool[req.table_idx, 11:13].tolist() == [21, 22]
    assert pool[req.table_idx, 10].item() == 0  # anchor input untouched


# ======================================================================================
# propose_drafts: the autoregressive loop over an injected step_fn (stub here)
# ======================================================================================


def _scripted_step(script: dict):
    """step_fn stub: the last row one-hots the token scripted for this prefix length."""

    def step_fn(prefix: torch.Tensor) -> torch.Tensor:
        n = prefix.numel()
        rows = [_onehot(0) * 0 for _ in range(n - 1)] + [_onehot(script[n - 1])]
        return torch.stack(rows)

    return step_fn


def test_propose_drafts_follows_the_stub_chain():
    tokens, logits = propose_drafts(_scripted_step({0: 3, 1: 1, 2: 5}), seed_token=9, depth=3)
    assert tokens.tolist() == [3, 1, 5]
    assert logits.shape == (3, V)
    assert logits.argmax(-1).tolist() == [3, 1, 5]


def test_propose_drafts_empty_at_zero_depth():
    tokens, logits = propose_drafts(_scripted_step({}), seed_token=9, depth=0)
    assert tokens.shape == (0,)
    assert logits.shape == (0, 0)


def test_propose_drafts_custom_chooser():
    # a chooser that always picks token 2: the rows still record the stub's script
    tokens, logits = propose_drafts(
        _scripted_step({0: 3, 1: 4}), seed_token=9, depth=2, choose=lambda row: 2)
    assert tokens.tolist() == [2, 2]
    assert logits.argmax(-1).tolist() == [3, 4]


# ======================================================================================
# verify_and_finalize: verdict -> commit -> accounting, against a real CacheManager
# ======================================================================================


def _reserve_with_pages(req: Req, cm: CacheManager, table_manager: TableManager):
    req.table_idx = table_manager.allocate()
    cm.allocate_paged([req])


def test_verify_and_finalize_full_accept_greedy():
    table_manager, cm = _managers()
    req = _req(cached=8, depth=2)
    _reserve_with_pages(req, cm, table_manager)
    free_before = len(cm.free_slots)
    accounting = SpecAccounting()
    gen = torch.Generator().manual_seed(0)
    verdict = verify_and_finalize(
        req, draft_tokens=torch.tensor([3, 1]),
        draft_logits=torch.stack([_onehot(3), _onehot(1)]),
        target_logits=torch.stack([_onehot(3), _onehot(1), _onehot(4)]),
        base_len=8, cache_manager=cm, accounting=accounting, generator=gen)
    assert verdict.all_accepted and verdict.n_accepted == 2
    assert not verdict.needs_replay
    assert verdict.accepted == [3, 1, 4]
    # host tail rewritten with the verdict, lengths rotated past it
    assert req.input_ids[9:12].tolist() == [3, 1, 4]
    assert (req.cached_len, req.device_len, req.spec_depth) == (11, 12, 0)
    # full accept keeps every reserved page: no tail to release
    assert len(cm.free_slots) == free_before
    assert accounting.snapshot() == {"steps": 1, "proposed": 2, "accepted": 2, "rate": 1.0}


def test_verify_and_finalize_reject_resamples_and_releases_once():
    table_manager, cm = _managers()
    req = _req(cached=8, depth=5)  # device 14: pages 2,3 allocated
    _reserve_with_pages(req, cm, table_manager)
    free_before = len(cm.free_slots)
    accounting = SpecAccounting()
    policy = AdaptiveDepthPolicy(5)
    gen = torch.Generator().manual_seed(0)
    verdict = verify_and_finalize(
        req, draft_tokens=torch.tensor([3, 1]),
        draft_logits=torch.stack([_onehot(3), _onehot(1)]),
        # row 0 rejects the draft (target argmax 5), so the resample replaces it
        target_logits=torch.stack([_onehot(5), _onehot(1), _onehot(4)]),
        base_len=8, cache_manager=cm, accounting=accounting,
        policy=policy, generator=gen)
    assert not verdict.all_accepted and verdict.n_accepted == 0
    assert verdict.accepted == [5] and verdict.committed == 9
    assert verdict.needs_replay  # [8, 9): the accepted-but-unplayed span
    assert req.input_ids[9].item() == 5
    assert (req.cached_len, req.device_len, req.spec_depth) == (9, 10, 0)
    # single-shot tail release: exactly the whole page past keep_len=9 comes back
    assert len(cm.free_slots) == free_before + 1
    assert accounting.snapshot() == {"steps": 1, "proposed": 2, "accepted": 0, "rate": 0.0}
    assert policy.depth_for(req.uid) == 4  # all-reject shrinks the adaptive depth


def test_verify_and_finalize_none_when_no_drafts():
    table_manager, cm = _managers()
    req = _req(cached=8, depth=0)
    _reserve_with_pages(req, cm, table_manager)
    accounting = SpecAccounting()
    verdict = verify_and_finalize(
        req, draft_tokens=torch.empty(0, dtype=torch.int64),
        draft_logits=torch.empty(0, 0), target_logits=torch.empty(1, V),
        base_len=8, cache_manager=cm, accounting=accounting)
    assert verdict is None
    assert (req.cached_len, req.device_len) == (8, 9)
    assert accounting.snapshot()["steps"] == 0


# ======================================================================================
# GPU-only seams raise instead of faking
# ======================================================================================


@pytest.mark.skipif(torch.cuda.is_available(), reason="CPU-only error-path test")
def test_draft_forward_needs_a_gpu():
    head = SimpleNamespace(draft_layer_ids=lambda: [48])
    with pytest.raises(RuntimeError, match="needs a GPU"):
        draft_forward(head, torch.tensor([1]), torch.zeros(1, 8), None)


# ======================================================================================
# CLI flags: --mtp-depth (default off) + --speculative-adaptive
# ======================================================================================


class _Config:
    def __init__(self, data: dict) -> None:
        self._data = data

    def to_dict(self) -> dict:
        return self._data


def _parse(argv: list) -> tuple:
    from freetoken.server.args import parse_args

    config = _Config({"architectures": ["Qwen4ExpForConditionalGeneration"],
                      "torch_dtype": "bfloat16"})
    with patch("freetoken.utils.cached_load_hf_config", lambda _path: config):
        return parse_args(["--model", "/models/anon", *argv])


def test_cli_mtp_defaults_off():
    args, _ = _parse([])
    assert args.mtp_depth == 0
    assert args.mtp_adaptive is False


def test_cli_mtp_flags_set():
    args, _ = _parse(["--mtp-depth", "3", "--speculative-adaptive"])
    assert args.mtp_depth == 3
    assert args.mtp_adaptive is True


def test_cli_adaptive_without_depth_is_rejected():
    with pytest.raises(SystemExit):
        _parse(["--speculative-adaptive"])


def test_cli_negative_depth_is_rejected():
    with pytest.raises(SystemExit):
        _parse(["--mtp-depth", "-1"])


# ======================================================================================
# EngineConfig: mtp override onto qwen4_args (num_nextn_predict_layers may be absent)
# ======================================================================================


def _hf_config(**text_overrides):
    text = SimpleNamespace(
        num_hidden_layers=4, hidden_size=128, vocab_size=512, head_dim=64,
        num_attention_heads=4, num_key_value_heads=1,
        layer_types=["linear_attention", "linear_attention", "linear_attention",
                     "full_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                         "partial_rotary_factor": 0.25},
        max_position_embeddings=4096, rms_norm_eps=1e-6, hidden_act="silu",
        tie_word_embeddings=False, num_experts=8, num_experts_per_tok=2,
        moe_intermediate_size=64, shared_expert_intermediate_size=64,
        norm_topk_prob=True, linear_num_key_heads=2, linear_num_value_heads=6,
        linear_key_head_dim=32, linear_value_head_dim=32, linear_conv_kernel_dim=4,
        output_gate_type="sigmoid", indexer_n_heads=2, indexer_kv_heads=1,
        indexer_head_dim=64, indexer_budget=16, indexer_compress_ratio=4,
        hc_count=4, hc_lowrank=16, ple_layer_ids=[2], ple_embed_dim=64,
        ple_conv_kernel_size=4, ngram_size=3, heads_per_ngram=2,
        ngram_vocab_size_base=1000, make_ngram_vocab_size_divisible_by=8,
        split_ngram_parts=4, eos_token_id=7,
    )
    for name, value in text_overrides.items():
        setattr(text, name, value)
    return SimpleNamespace(model_type="qwen4_exp",
                           architectures=["Qwen4ExpForConditionalGeneration"],
                           text_config=text, quantization_config=None)


def test_apply_mtp_override_passthrough_when_off():
    from freetoken.models.qwen4_exp.config import parse_config

    cfg = parse_config(_hf_config())
    assert cfg.qwen4_args.mtp_num_layers == 0
    assert apply_mtp_override(cfg, 0) is cfg


def test_apply_mtp_override_falls_back_to_one_layer():
    from freetoken.models.qwen4_exp.config import parse_config

    cfg = parse_config(_hf_config())  # HF configs carry no num_nextn_predict_layers
    out = apply_mtp_override(cfg, 2)
    assert out.qwen4_args.mtp_num_layers == 1
    assert cfg.qwen4_args.mtp_num_layers == 0  # input untouched (frozen replace)


def test_apply_mtp_override_keeps_parsed_layers():
    from freetoken.models.qwen4_exp.config import parse_config

    cfg = parse_config(_hf_config(num_nextn_predict_layers=2))
    assert apply_mtp_override(cfg, 3) is cfg


def test_apply_mtp_override_rejects_non_mtp_models():
    with pytest.raises(ValueError, match="qwen4_exp"):
        apply_mtp_override(SimpleNamespace(model_type="qwen3", qwen4_args=None), 2)


def test_engine_config_rejects_negative_depth():
    with pytest.raises(ValueError, match="mtp_depth"):
        EngineConfig(model_path="x", tp_info=DistributedInfo(0, 1),
                     dtype=torch.bfloat16, mtp_depth=-1)


# ======================================================================================
# SpecAccounting: counters, rate, snapshot shape (the 419.8 contract)
# ======================================================================================


def test_accounting_idle_rate_and_fragment():
    accounting = SpecAccounting()
    assert accounting.rate == 0.0
    assert accounting.log_fragment() == ""
    assert accounting.snapshot() == {"steps": 0, "proposed": 0, "accepted": 0, "rate": 0.0}


def test_accounting_record_and_fragment():
    accounting = SpecAccounting()
    accounting.record(n_accepted=2, depth=2)
    accounting.record(n_accepted=1, depth=3)
    assert accounting.snapshot() == {"steps": 2, "proposed": 5, "accepted": 3, "rate": 0.6}
    assert accounting.log_fragment() == "spec accept: 3/5 (0.60)"


def test_accounting_rejects_impossible_counts():
    with pytest.raises(AssertionError):
        SpecAccounting().record(n_accepted=3, depth=2)


# ======================================================================================
# Status line: acceptance rate next to throughput; plain lines byte-identical
# ======================================================================================


def _reporter(interval=1):
    logs: list[str] = []
    clock = {"t": 0.0}
    rep = SchedulerStatusReporter(log=logs.append, clock=lambda: clock["t"],
                                  decode_log_interval=interval)
    return rep, logs, clock


def _decode_batch(n):
    reqs = [SimpleNamespace(extend_len=1, cached_len=0) for _ in range(n)]
    return SimpleNamespace(is_prefill=False, is_decode=True, reqs=reqs)


def test_decode_line_reports_acceptance_next_to_throughput():
    rep, logs, clock = _reporter()
    clock["t"] = 2.0
    rep.report_batch(
        _decode_batch(2), running_reqs=2, queue_reqs=0,
        kv_used_pages=60, kv_total_pages=200, page_size=16,
        spec={"steps": 2, "proposed": 4, "accepted": 3, "rate": 0.75})
    line = logs[-1]
    assert "gen throughput (token/s)" in line
    assert "spec accept: 3/4 (0.75)" in line
    # the fragment sits between throughput and the queue count
    assert line.index("spec accept") > line.index("gen throughput")
    assert line.index("spec accept") < line.index("#queue-req")


def test_decode_line_byte_identical_without_spec():
    for spec in (None, {"steps": 0, "proposed": 0, "accepted": 0, "rate": 0.0}):
        rep, logs, clock = _reporter()
        clock["t"] = 2.0
        rep.report_batch(_decode_batch(2), running_reqs=2, queue_reqs=0,
                         kv_used_pages=60, kv_total_pages=200, page_size=16, spec=spec)
        assert "spec" not in logs[-1]
    assert _spec_msg(None) == ""
    assert _spec_msg({"steps": 0, "proposed": 0, "accepted": 0, "rate": 0.0}) == ""


def test_decode_throughput_counts_the_accepted_span():
    rep, logs, clock = _reporter()
    batch = _decode_batch(1)
    # one request verified 1 draft + bonus: 3 generated tokens, not 1
    batch.spec_accepted = {0: [11, 12, 13]}
    clock["t"] = 3.0  # 3 tokens over 3.0s -> 1 tok/s; plain counting would read 0.33
    rep.report_batch(batch, running_reqs=1, queue_reqs=0,
                     kv_used_pages=1, kv_total_pages=10, page_size=1)
    assert "gen throughput (token/s): 1.00" in logs[-1]


# ======================================================================================
# Batch defaults: plain batches carry no speculative markers
# ======================================================================================


def test_batch_spec_markers_default_off():
    batch = Batch(reqs=[], phase="decode")
    assert batch.spec_rows is None
    assert batch.spec_finalized is False
    assert batch.spec_accepted is None
    assert not batch.spec_active


# ======================================================================================
# MTP expert banks: the consumer iter_mtp_expert_pieces was missing (419.2 note)
# ======================================================================================


def _mtp_bank_config():
    return SimpleNamespace(num_experts=2, moe_intermediate_size=128, hidden_size=128,
                           num_moe_layers=48)


def _write_mtp_fake_ckpt(folder: str):
    from safetensors.torch import save_file

    E, I, H = 2, 128, 128
    base = "mtp.layers.0.mlp.experts"
    tensors = {
        f"{base}.gate_up_proj": torch.randn(E, 2 * I, H, dtype=torch.bfloat16),
        f"{base}.down_proj": torch.randn(E, H, I, dtype=torch.bfloat16),
    }
    import os

    save_file(tensors, os.path.join(folder, "model.safetensors"))


def test_collect_mtp_expert_pieces_keys_and_bank_ids(tmp_path):
    _write_mtp_fake_ckpt(str(tmp_path))
    pieces = collect_mtp_expert_pieces(str(tmp_path), _mtp_bank_config())
    assert len(pieces) == 1  # E=2 fits one batch of 32
    layer_id, e0, e1, part = pieces[0]
    assert (layer_id, e0, e1) == (48, 0, 2)  # past the target's MoE layers
    assert set(part) == {"gate", "up", "down"}
    assert tuple(part["gate"].shape) == (2, 128, 128)
    assert tuple(part["up"].shape) == (2, 128, 128)
    assert tuple(part["down"].shape) == (2, 128, 128)
    assert part["gate"].dtype is torch.bfloat16


def test_mtp_bank_layer_ids_match_the_head():
    assert mtp_bank_layer_ids(_mtp_bank_config(), 1) == [48]
    assert mtp_bank_layer_ids(_mtp_bank_config(), 2) == [48, 49]
    assert mtp_bank_layer_ids(_mtp_bank_config(), 0) == []


def test_draft_qsa_layer_ids_follow_the_bank_ids():
    from freetoken.engine.mtp import draft_qsa_layer_ids

    assert draft_qsa_layer_ids(_mtp_bank_config()) == []
    with_mtp = SimpleNamespace(num_moe_layers=48, qwen4_args=SimpleNamespace(mtp_num_layers=1))
    assert draft_qsa_layer_ids(with_mtp) == [48]
    with_two = SimpleNamespace(num_moe_layers=48, qwen4_args=SimpleNamespace(mtp_num_layers=2))
    assert draft_qsa_layer_ids(with_two) == [48, 49]
    zero = SimpleNamespace(num_moe_layers=48, qwen4_args=SimpleNamespace(mtp_num_layers=0))
    assert draft_qsa_layer_ids(zero) == []


def test_draft_moe_layers_empty_without_head():
    from types import SimpleNamespace

    from freetoken.engine.mtp import draft_moe_layers

    assert draft_moe_layers(SimpleNamespace(mtp=None)) == []


# ======================================================================================
# Anchor hidden plumbing: per-request anchor rows out of the stashed residual
# ======================================================================================


def _anchor_req(uid: int, extend: int = 1, cached: int = 10) -> SimpleNamespace:
    return SimpleNamespace(uid=uid, extend_len=extend, cached_len=cached)


def test_spec_anchor_hidden_none_without_a_model_residual():
    from freetoken.engine.mtp import spec_anchor_hidden

    batch = SimpleNamespace(spec_active=False, is_prefill=False, reqs=[])
    assert spec_anchor_hidden(None, batch) is None


def test_spec_anchor_hidden_retains_each_requests_span():
    """The draft closure pairs (h_{p-1}, x_p), so the whole span of rows is kept
    with its base position -- the accepted row count is only known after verify."""
    from freetoken.engine.mtp import spec_anchor_hidden

    r1, r2 = _anchor_req(1, cached=20), _anchor_req(7, cached=50)
    batch = SimpleNamespace(
        spec_active=True, spec_rows=[(r1, 0), (r1, 1), (r1, 2), (r2, 0), (r2, 1)])
    hidden = torch.arange(5 * 4, dtype=torch.float32).view(5, 4)
    anchors = spec_anchor_hidden(hidden, batch)
    assert set(anchors) == {1, 7}
    base1, span1 = anchors[1]
    base7, span7 = anchors[7]
    assert base1 == 20 and torch.equal(span1, hidden[0:3])
    assert base7 == 50 and torch.equal(span7, hidden[3:5])


def test_spec_anchor_hidden_skips_the_padding_uid():
    from freetoken.engine.mtp import spec_anchor_hidden

    dummy = _anchor_req(-1)
    batch = SimpleNamespace(spec_active=True, spec_rows=[(dummy, 0)])
    anchors = spec_anchor_hidden(torch.zeros(1, 4), batch)
    assert anchors == {}


def test_spec_anchor_hidden_takes_chunk_tails_of_a_prefill():
    from freetoken.engine.mtp import spec_anchor_hidden

    r1, r2 = _anchor_req(1, 3, cached=0), _anchor_req(2, 2, cached=0)
    batch = SimpleNamespace(spec_active=False, is_prefill=True, padded_reqs=[r1, r2])
    hidden = torch.arange(5 * 4, dtype=torch.float32).view(5, 4)
    anchors = spec_anchor_hidden(hidden, batch)
    base1, span1 = anchors[1]
    base2, span2 = anchors[2]
    assert base1 == 0 and torch.equal(span1, hidden[0:3])
    assert base2 == 0 and torch.equal(span2, hidden[3:5])


def test_spec_anchor_hidden_takes_plain_decode_rows():
    from freetoken.engine.mtp import spec_anchor_hidden

    r1, r2 = _anchor_req(1, cached=31), _anchor_req(2, cached=7)
    batch = SimpleNamespace(
        spec_active=False, is_prefill=False, padded_reqs=[r1, r2])
    hidden = torch.arange(2 * 4, dtype=torch.float32).view(2, 4)
    anchors = spec_anchor_hidden(hidden, batch)
    base1, span1 = anchors[1]
    base2, span2 = anchors[2]
    assert base1 == 31 and torch.equal(span1, hidden[0:1])
    assert base2 == 7 and torch.equal(span2, hidden[1:2])


def test_draft_forward_needs_a_gpu_not_hidden_states():
    from freetoken.engine.mtp import draft_forward

    with pytest.raises(RuntimeError, match="needs a GPU"):
        draft_forward(object(), torch.zeros(1), torch.zeros(1, 4), None)


def test_forward_output_anchor_field_defaults_to_none():
    from freetoken.engine.engine import ForwardOutput

    out = ForwardOutput(object(), object(), object())
    assert out.spec_logits is None and out.target_hidden_anchors is None
    anchors = {1: torch.zeros(4)}
    out2 = ForwardOutput(object(), object(), object(), None, anchors)
    assert out2.target_hidden_anchors is anchors


def test_qsa_ring_capacity_widens_with_depth():
    from freetoken.kvcache.qsa_pool import QSAKVCache

    assert qsa_ring_capacity_for_depth(4, 0) == QSAKVCache.ring_capacity_for(4)
    assert qsa_ring_capacity_for_depth(4, 2) == QSAKVCache.ring_capacity_for(4, 2) > (
        QSAKVCache.ring_capacity_for(4))


# ======================================================================================
# Loader switch: include_mtp reaches only readers that take it
# ======================================================================================


def test_load_weight_forwards_include_mtp_when_supported(monkeypatch):
    import freetoken.models.weight as loader

    seen: dict = {}

    def fake_iter(model_path, device, **kwargs):
        seen.update(kwargs)
        return iter(())

    spec = SimpleNamespace(module="fake.module", iter_weights="iter_weights")
    monkeypatch.setattr(loader, "_spec_for_model_path", lambda path: (None, spec))
    monkeypatch.setattr(loader, "_load_attr", lambda module, name: fake_iter)
    monkeypatch.setattr("freetoken.checkpoint.ftw.is_ftw_checkpoint", lambda path: False)
    list(loader.load_weight("ckpt", torch.device("cpu"), include_mtp=True))
    assert seen == {"include_moe_experts": True, "include_non_moe": True,
                    "include_mtp": True}


def test_load_weight_ignores_include_mtp_for_other_models(monkeypatch):
    import freetoken.models.weight as loader

    seen: dict = {}

    def fake_iter(model_path, device, include_moe_experts, include_non_moe):
        seen["include_moe_experts"] = include_moe_experts
        return iter(())

    spec = SimpleNamespace(module="fake.module", iter_weights="iter_weights")
    monkeypatch.setattr(loader, "_spec_for_model_path", lambda path: (None, spec))
    monkeypatch.setattr(loader, "_load_attr", lambda module, name: fake_iter)
    monkeypatch.setattr("freetoken.checkpoint.ftw.is_ftw_checkpoint", lambda path: False)
    list(loader.load_weight("ckpt", torch.device("cpu"), include_mtp=True))
    assert seen == {"include_moe_experts": True}
