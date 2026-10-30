"""Механика хуков на двух архитектурах: Llama и Qwen3.5 (гибрид линейного и
полного внимания, как у Qwen3.x). Модели собираются из конфига, без скачивания.

Прогнать ДО любых экспериментов: молча не сработавший хук даёт «нулевой эффект»,
который легко принять за опровержение H3.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from steering_tool_feedback import simulator as S  # noqa: E402
from steering_tool_feedback import steering as ST  # noqa: E402
from steering_tool_feedback.windows import PositionWindow, prepare_prompt  # noqa: E402

SEQ = 24


def _inputs(model, seq=SEQ, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(5, model.config.vocab_size, (1, seq), generator=g)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def _vec(model, layer=None):
    layer = ST.n_layers(model) // 2 if layer is None else layer
    return ST.SteeringVector(layer, np.ones(model.config.hidden_size, dtype=np.float32),
                             "post_message", {"test": True})


@torch.no_grad()
def _logits(model, batch):
    return model(**batch).logits.clone()


# --------------------------------------------------------------------------


def test_layers_found(tiny_model):
    _, m = tiny_model
    assert ST.n_layers(m) == 4


def test_recorder(tiny_model):
    _, m = tiny_model
    b = _inputs(m)
    with ST.ActivationRecorder(m, layers=[1, 3]) as rec:
        with torch.no_grad():
            m(**b)
    acts = rec.pooled(PositionWindow(4, 10, "post_message"))
    assert set(acts) == {1, 3}
    assert acts[1].shape == (m.config.hidden_size,)
    assert np.isfinite(acts[1]).all()
    assert rec.raw(3).shape == (SEQ, m.config.hidden_size)
    with pytest.raises(ValueError):
        rec.pooled(PositionWindow(0, SEQ + 5, "post_message"))


def test_hook_fires_exactly_in_window(tiny_model):
    name, m = tiny_model
    b = _inputs(m)
    clean = _logits(m, b)
    w = PositionWindow(10, 14, "post_message")
    with ST.steering(m, _vec(m), ST.SteerConfig(alpha=5.0, window=w)) as h:
        steered = _logits(m, b)
    assert h.n_applied == len(w)
    assert h.positions_source in ("cache_position", "position_ids"), (
        f"{name}: positions came from a counter, not from the model"
    )
    # Причинность: до окна ничего не меняется, начиная с окна — меняется.
    assert torch.allclose(clean[:, : w.start], steered[:, : w.start], atol=1e-5)
    assert not torch.allclose(clean[:, w.start:], steered[:, w.start:])


def test_noops(tiny_model):
    _, m = tiny_model
    b = _inputs(m)
    clean = _logits(m, b)
    w = PositionWindow(10, 14, "post_message")
    with ST.steering(m, _vec(m), ST.SteerConfig(alpha=0.0, window=w)) as h0:
        z = _logits(m, b)
    with ST.steering(m, _vec(m), ST.SteerConfig(alpha=5.0, window=w, enabled=False)) as h1:
        d = _logits(m, b)
    assert torch.allclose(clean, z, atol=1e-6) and h0.n_applied == 0
    assert torch.allclose(clean, d, atol=1e-6) and h1.n_applied == 0


def test_detach_restores(tiny_model):
    _, m = tiny_model
    b = _inputs(m)
    clean = _logits(m, b)
    with ST.steering(m, _vec(m), ST.SteerConfig(alpha=5.0, window=PositionWindow(3, 9))):
        _logits(m, b)
    assert torch.allclose(clean, _logits(m, b), atol=1e-6)


def test_several_sequences_in_one_context(tiny_model):
    """Регрессия: старый счётчик позиций уезжал на втором прогоне в том же контексте."""
    _, m = tiny_model
    w = PositionWindow(10, 14, "post_message")
    with ST.steering(m, _vec(m), ST.SteerConfig(alpha=5.0, window=w)) as h:
        _logits(m, _inputs(m, seed=1))
        _logits(m, _inputs(m, seed=2))
        _logits(m, _inputs(m, seq=30, seed=3))
    assert h.n_sequences == 3
    assert h.n_applied == 3 * len(w)


def test_generate_prefill_only_by_default(tiny_model):
    _, m = tiny_model
    b = _inputs(m)
    w = PositionWindow(10, 14, "post_message")
    with ST.steering(m, _vec(m), ST.SteerConfig(alpha=3.0, window=w)) as h:
        with torch.no_grad():
            out = m.generate(**b, max_new_tokens=5, min_new_tokens=5, do_sample=False)
    assert out.shape[1] == SEQ + 5
    assert h.n_applied == len(w), "steered generated tokens although apply_to_generated=False"


def test_generate_apply_to_generated(tiny_model):
    _, m = tiny_model
    b = _inputs(m)
    w = PositionWindow(10, 14, "post_message", apply_to_generated=True)
    new = 5
    with ST.steering(m, _vec(m), ST.SteerConfig(alpha=3.0, window=w)) as h:
        with torch.no_grad():
            m.generate(**b, max_new_tokens=new, min_new_tokens=new, do_sample=False)
    # prefill даёт 1-й токен, ещё (new - 1) шагов декодирования по одной позиции.
    assert h.n_applied == len(w) + (new - 1)


def test_generate_kv_cache_consistent(tiny_model):
    """С KV-кэшем и без — одинаковый результат под вмешательством."""
    _, m = tiny_model
    b = _inputs(m)
    w = PositionWindow(10, 14, "post_message")
    outs = []
    for use_cache in (True, False):
        with ST.steering(m, _vec(m), ST.SteerConfig(alpha=3.0, window=w)):
            with torch.no_grad():
                outs.append(m.generate(**b, max_new_tokens=6, min_new_tokens=6,
                                       do_sample=False, use_cache=use_cache))
    assert torch.equal(outs[0], outs[1])


def test_batch_gt_one_rejected(tiny_model):
    _, m = tiny_model
    ids = torch.randint(5, m.config.vocab_size, (2, SEQ))
    with ST.steering(m, _vec(m), ST.SteerConfig(alpha=1.0, window=PositionWindow(2, 5))):
        with pytest.raises(RuntimeError, match="batch=1"):
            with torch.no_grad():
                m(input_ids=ids)


def test_dose_response_monotone(tiny_model):
    _, m = tiny_model
    b = _inputs(m)
    clean = _logits(m, b)
    w = PositionWindow(10, 14, "post_message")

    def dist(a):
        with ST.steering(m, _vec(m), ST.SteerConfig(alpha=a, window=w)):
            return float((_logits(m, b) - clean).norm())

    d = [x for _, x in ST.dose_response([0.0, 0.5, 2.0, 8.0], dist)]
    assert d[0] == 0.0
    assert all(d[i] < d[i + 1] for i in range(len(d) - 1)), d


def test_sham_is_a_fair_control(tiny_model):
    _, m = tiny_model
    b = _inputs(m)
    clean = _logits(m, b)
    w = PositionWindow(10, 14, "post_message")
    real, sham = _vec(m), ST.random_direction_like(_vec(m), seed=0)
    with ST.steering(m, real, ST.SteerConfig(alpha=5.0, window=w)) as hr:
        dr = float((_logits(m, b) - clean).norm())
    with ST.steering(m, sham, ST.SteerConfig(alpha=5.0, window=w)) as hs:
        ds = float((_logits(m, b) - clean).norm())
    assert hr.n_applied == hs.n_applied == len(w)
    assert ds > 0 and dr > 0


def test_vector_save_load(tmp_path):
    v = ST.SteeringVector(3, np.arange(8, dtype=np.float32), "field_read", {"seed": 1, "src": "t"})
    path = tmp_path / "v.npz"
    v.save(str(path))
    u = ST.SteeringVector.load(str(path))
    assert u.layer == 3 and u.window_kind == "field_read"
    assert np.allclose(u.vector, v.vector) and u.provenance == v.provenance


# --------------------------------------------------------------------------
# Сквозная проверка: реальный шаблон + окна + хук на архитектуре Qwen3.5
# --------------------------------------------------------------------------


@pytest.mark.real_tokenizer
def test_end_to_end_real_template(real_tokenizer):
    from conftest import make_tiny_qwen35

    m = make_tiny_qwen35(vocab=len(real_tokenizer))
    t = next(x for x in S.build_pilot_set(60, seed=0) if x.pair_type == "changing")
    p = prepare_prompt(real_tokenizer, t.prompt(1), essential_field=t.essential_field(1),
                       tools=S.tools_schema())
    b = p.model_inputs()

    # Не логиты: при словаре 248k полные логиты на ~1k токенов ≈ 1 ГБ fp32,
    # несколько копий убивают процесс по памяти. Причинность видна и по
    # последнему скрытому состоянию базовой модели.
    @torch.no_grad()
    def hidden():
        return m.model(**b).last_hidden_state.clone()

    clean = hidden()
    for kind in ("field_read", "post_message", "pre_action"):
        w = p.windows[kind]
        with ST.steering(m, _vec(m), ST.SteerConfig(alpha=5.0, window=w)) as h:
            steered = hidden()
        assert h.n_applied == len(w), kind
        assert h.positions_source in ("cache_position", "position_ids"), kind
        assert torch.allclose(clean[:, : w.start], steered[:, : w.start], atol=1e-5), kind
        assert not torch.allclose(clean[:, w.start:], steered[:, w.start:]), kind


# --------------------------------------------------------------------------
# TACT: проекция на ось с полосой (ProjectionHook)
# --------------------------------------------------------------------------


@torch.no_grad()
def _hidden(model, batch, layer):
    with ST.ActivationRecorder(model, [layer]) as rec:
        model(**batch)
    return rec.raw(layer)


def _axis_through(model, batch, layer, pos, sigma):
    """Ось, на которой состояние в позиции pos лежит ровно на 3·sigma от центра."""
    h = _hidden(model, batch, layer)[pos]
    rng = np.random.default_rng(0)
    u = rng.normal(size=h.shape)
    u /= np.linalg.norm(u)
    mu = h - 3.0 * sigma * u
    return ST.DriftAxis(layer, u.astype(np.float32), mu.astype(np.float32), sigma)


def test_projection_hook_clamps_outside_band_only(tiny_model):
    _, m = tiny_model
    b = _inputs(m)
    L, pos = 2, SEQ - 1
    w = PositionWindow(pos, pos + 1, "action_point")
    ax = _axis_through(m, b, L, pos, sigma=0.5)
    clean = _logits(m, b)
    # k = 2: проекция 3σ вне полосы 2σ -> сдвиг; lam = 1 -> ровно на границу полосы.
    with ST.drift_clamp(m, [ax], w, k=2.0, lam=1.0) as g:
        with ST.ActivationRecorder(m, [L]) as rec:
            steered = m(**b).logits.clone()
    assert g.n_applied == 1 and g.n_changed == 1
    assert g.positions_source in ("cache_position", "position_ids")
    p_after = float((rec.raw(L)[pos] - ax.mu) @ ax.unit)
    assert abs(p_after - 2.0 * ax.sigma) < 1e-3, p_after
    assert torch.allclose(clean[:, :pos], steered[:, :pos], atol=1e-5)
    assert not torch.allclose(clean[:, pos:], steered[:, pos:])
    # k = 4: проекция 3σ внутри полосы -> ничего не меняется.
    with ST.drift_clamp(m, [ax], w, k=4.0, lam=1.0) as g2:
        inside = _logits(m, b)
    assert g2.n_applied == 1 and g2.n_changed == 0
    assert torch.allclose(clean, inside, atol=1e-6)
    # lam = 0 — no-op.
    with ST.drift_clamp(m, [ax], w, k=2.0, lam=0.0) as g3:
        zero = _logits(m, b)
    assert g3.n_applied == 0 and torch.allclose(clean, zero, atol=1e-6)


def test_projection_hook_several_layers(tiny_model):
    _, m = tiny_model
    b = _inputs(m)
    pos = SEQ - 1
    axes = [_axis_through(m, b, L, pos, 0.5) for L in (1, 2)]
    with ST.drift_clamp(m, axes, PositionWindow(pos, pos + 1, "action_point"), 2.0, 1.0) as g:
        _logits(m, b)
    assert g.n_applied == 1 and len(g.hooks) == 2 and all(h.n_applied == 1 for h in g.hooks)


def test_drift_axis_per_problem():
    rng = np.random.default_rng(0)
    d = rng.normal(size=16)
    d /= np.linalg.norm(d)
    ok = [rng.normal(size=16) * 0.1 + i for i in range(6)]          # сдвиг по задачам
    fail = [a + 2 * d for a in ok]
    groups = [f"t{i}" for i in range(6)]
    ax, info = ST.drift_axis(fail, ok, groups, groups, layer=3)
    assert info["mode"] == "per_problem" and info["n_problems"] == 6
    assert float(ax.unit @ d) > 0.99, "ось = разность внутри задач, задачи не мешают"
    ax2, info2 = ST.drift_axis(fail, ok, ["a"] * 6, ["b"] * 6, layer=3)
    assert info2["mode"] == "global_means"
    with pytest.raises(ValueError):
        ST.drift_axis([], ok, [], groups, 3)


# --------------------------------------------------------------------------
# Подстановка активаций (диагностика)
# --------------------------------------------------------------------------


def test_patch_hook_replaces_exactly_the_window(tiny_model):
    _, m = tiny_model
    b = _inputs(m)
    clean = _logits(m, b)
    L, w = 1, PositionWindow(6, 11, "tool_content")
    h = _hidden(m, b, L)
    # подстановка собственных значений — ничего не меняет
    with ST.patched(m, L, w, h[w.start:w.end]) as hook:
        same = _logits(m, b)
    assert hook.n_applied == len(w) and torch.allclose(clean, same, atol=1e-5)
    # чужие значения: до окна всё как было, с окна — иначе; в слое L окно стало ровно тем, что подставили
    other = np.random.default_rng(0).normal(size=(len(w), m.config.hidden_size)).astype(np.float32)
    with ST.patched(m, L, w, other):
        with ST.ActivationRecorder(m, [L]) as rec:
            changed = m(**b).logits.clone()
    assert np.allclose(rec.raw(L)[w.start:w.end], other, atol=1e-3)
    assert torch.allclose(clean[:, : w.start], changed[:, : w.start], atol=1e-5)
    assert not torch.allclose(clean[:, w.start:], changed[:, w.start:])
    with pytest.raises(ValueError):
        ST.PatchHook(L, w, other[:2])


def test_patch_last_layer_reproduces_donor_logits(tiny_model):
    """Подстановка ВСЕХ позиций на последнем слое = логиты донора (дальше только норма и голова)."""
    name, m = tiny_model
    import copy as _copy

    donor = _copy.deepcopy(m)
    with torch.no_grad():
        for p in ST.get_layers(donor)[-1].parameters():
            p.add_(0.05 * torch.randn_like(p))
    b = _inputs(m)
    last = ST.n_layers(m) - 1
    want = _logits(donor, b)
    hd = _hidden(donor, b, last)
    with ST.patched(m, last, PositionWindow(0, SEQ, "tool_content"), hd):
        got = _logits(m, b)
    assert torch.allclose(want, got, atol=1e-4), name


def test_affine_hook_exact(tiny_model):
    _, m = tiny_model
    b = _inputs(m)
    clean = _logits(m, b)
    d = m.config.hidden_size
    rng = np.random.default_rng(0)
    U, P, a0 = rng.normal(size=(d, 2)) * 0.05, rng.normal(size=(2, d)), rng.normal(size=d)
    L, w = 2, PositionWindow(SEQ - 1, SEQ, "action_point")
    with ST.affine(m, L, w, U, P, a0, alpha=0.0) as h0:
        z = _logits(m, b)
    assert h0.n_applied == 1 and torch.allclose(clean, z, atol=1e-5)
    h_base = _hidden(m, b, L)[SEQ - 1]
    hook = ST.AffineHook(L, w, U, P, a0, alpha=0.7)
    hook.attach(m)
    try:
        with ST.ActivationRecorder(m, [L]) as rec:
            out = m(**b).logits.clone()
    finally:
        hook.detach()
    want = h_base + 0.7 * (h_base @ U @ P + a0)
    assert hook.n_applied == 1
    assert np.allclose(rec.raw(L)[SEQ - 1], want, atol=1e-3)
    assert torch.allclose(clean[:, : SEQ - 1], out[:, : SEQ - 1], atol=1e-5)
    with pytest.raises(ValueError):
        ST.AffineHook(L, w, U, P[:1], a0)


def test_affine_stack_matches_sequential_hooks(tiny_model):
    _, m = tiny_model
    b = _inputs(m)
    d = m.config.hidden_size
    rng = np.random.default_rng(1)
    w = PositionWindow(SEQ - 1, SEQ, "action_point")

    class Part:
        def __init__(self, layer, alpha):
            self.layer, self.window, self.alpha = layer, w, alpha
            self.U, self.P, self.a0 = rng.normal(size=(d, 2)) * 0.05, rng.normal(size=(2, d)), rng.normal(size=d)

    parts = [Part(1, 0.5), Part(3, 0.3)]
    with ST.affine_stack(m, parts) as g:
        got = _logits(m, b)
    assert g.n_applied == 1 and len(g.hooks) == 2
    hooks = [ST.AffineHook(p.layer, w, p.U, p.P, p.a0, p.alpha) for p in parts]
    for h in hooks:
        h.attach(m)
    try:
        want = _logits(m, b)
    finally:
        for h in hooks:
            h.detach()
    assert torch.allclose(got, want, atol=1e-5)
    assert not torch.allclose(got, _logits(m, b), atol=1e-4)
