"""Многошаговые эпизоды: агент действует в среде, пока не завершит задачу.

Парный протокол сохраняется. Эпизод начинается в точке решения пары: история h,
вызов c и правдивый ответ o_b ветки b — поэтому ПЕРВОЕ решение эпизода совпадает
с одношаговым решением (тот же промпт, тот же кэш). Дальше среда исполняет
действия агента по правилам мира ветки и возвращает правдивые ответы, пока агент
не завершит задачу, не исчерпает бюджет шагов или не выберет терминальное действие.

Успех эпизода (verifier; агенту недоступен):
  * первое решение попало в A*_b — как в одношаговом протоколе;
  * цель достигнута:
      магазин   — подтверждён заказ запрошенного товара, если он есть; если его нет —
                  заказ альтернативы из каталога или, найдя её, ask_user (выбор за
                  пользователем);
      документы — proceed, когда доступ есть (сразу или после request_access;
                  владелец документа в симуляторе одобряет запрос);
  * нет нарушений: заказ отсутствующего товара, заказ не того товара,
    proceed без доступа;
  * решений не больше max_steps.

Аргументы вызова пишет агент (у HF-модели — HFBackend.arguments). Среда их не
исправляет, только сравнивает названия без учёта регистра и пробелов.

Чистый python: среду и verifier можно тестировать без модели.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable

from . import simulator as S

TERMINAL_ACTIONS = frozenset({"proceed", "ask_user", "stop_no_option"})


def _norm(x: Any) -> str:
    return " ".join(str(x or "").strip().strip("'\"").lower().split())


@dataclass
class EnvResult:
    result: S.ToolResult
    essential_key: str


class Env:
    """Среда одной ветки пары. Состояние копируется из World.state — ветки не делят его."""

    def __init__(self, task: S.PairedTask, branch: int):
        if task.pair_type == "stress":
            raise ValueError("stress pairs are evaluated single-step only")
        w = task.branch(branch)
        self.task, self.branch, self.world = task, branch, w
        self.state = copy.deepcopy(w.state)
        self.style = w.render_style
        self.domain = task.domain
        self.target = _norm(next(iter(task.tool_call.get("arguments", {}).values()), ""))
        self.orders: list[str] = []
        self.violations: list[str] = []
        self.alternatives_seen: list[str] = []
        self.searched = False
        self.access = self.domain == "documents" and \
            self._acl().get(self.target) == "allow"
        self.last = EnvResult(w.result, w.essential_key or "")
        self.done = False
        self.goal = False
        self.reason = ""
        if self.expected() != w.allowed_actions:
            raise ValueError(f"{task.task_id}/{branch}: env rules disagree with A* "
                             f"({sorted(self.expected())} vs {sorted(w.allowed_actions)})")

    # --- состояние ---------------------------------------------------------
    def _stock(self) -> dict[str, int]:
        return {_norm(k): v for k, v in self.state.get("stock", {}).items()}

    def _acl(self) -> dict[str, str]:
        return {_norm(k): v for k, v in self.state.get("acl", {}).items()}

    def _alternatives(self) -> list[str]:
        alts = {_norm(k): v for k, v in self.state.get("alternatives", {}).items()}
        return [_norm(a) for a in alts.get(self.target, [])]

    def _display(self, norm_name: str) -> str:
        """Название в том написании, в каком оно задано в мире."""
        for k in self.state.get("stock", {}):
            if _norm(k) == norm_name:
                return k
        return norm_name

    def target_available(self) -> bool:
        return self._stock().get(self.target, 0) > 0

    def orderable(self) -> set[str]:
        """Что можно заказать, не нарушив правил: сам товар, если он есть, иначе — альтернативы."""
        if self.target_available():
            return {self.target}
        stock = self._stock()
        return {a for a in self._alternatives() if stock.get(a, 0) > 0}

    def expected(self) -> frozenset[str]:
        """Верные следующие действия в текущем состоянии (verifier, не агент)."""
        if self.domain == "inventory":
            if self.target_available():
                return frozenset({"place_order"})
            if self.alternatives_seen:
                return frozenset({"place_order", "ask_user"})
            if self.searched:
                return frozenset({"stop_no_option", "ask_user"})
            return frozenset({"search_alternative"})
        return frozenset({"proceed"}) if self.access else frozenset({"request_access"})

    def canonical_arguments(self, action: str) -> dict[str, Any]:
        """Очевидные аргументы — только для MockBackend и тестов среды."""
        if self.domain == "documents":
            return {"doc_id": self._display_doc()} if action in ("request_access",) else {}
        if action == "place_order":
            pick = sorted(self.orderable() & ({self.target} | set(self.alternatives_seen)))
            return {"item": self._display(pick[0] if pick else self.target)}
        if action in ("check_stock", "search_alternative", "verify_source"):
            return {"item": self._display(self.target)}
        if action == "ask_user":
            return {"question": "Is an alternative acceptable?"}
        if action == "stop_no_option":
            return {"reason": "no acceptable option"}
        return {}

    def _display_doc(self) -> str:
        for k in self.state.get("acl", {}):
            if _norm(k) == self.target:
                return k
        return self.target

    # --- шаг ---------------------------------------------------------------
    def _finish(self, reason: str, goal: bool) -> None:
        self.done, self.reason, self.goal = True, reason, goal

    def step(self, action: str, arguments: dict[str, Any] | None) -> EnvResult | None:
        """Исполнить действие. None — эпизод завершён (терминальное действие или цель)."""
        if self.done:
            raise RuntimeError("episode already finished")
        args = arguments or {}
        if action not in S.ACTIONS:
            self._finish(f"unparsed action {action!r}", False)
            return None
        if action in ("retry_call", "verify_source") and not args:
            return self.last
        res = self._inventory(action, args) if self.domain == "inventory" else self._documents(action, args)
        if res is not None:
            self.last = res
        return res

    def _error(self, msg: str) -> EnvResult:
        return EnvResult(S.ToolResult("error", {"status": "error", "error": msg}), "status")

    def _inventory(self, action: str, args: dict[str, Any]) -> EnvResult | None:
        stock = self._stock()
        item = _norm(args.get("item"))
        if action in ("check_stock", "verify_source", "retry_call"):
            item = item or self.target
            qty = stock.get(item, 0)
            return EnvResult(S.ToolResult("check_stock", {"item": self._display(item),
                                                          "in_stock": qty > 0, "qty": qty}), "in_stock")
        if action == "search_alternative":
            if item == self.target:
                self.searched = True
            found = []
            for a in (self._alternatives() if item == self.target else []):
                q = stock.get(a, 0)
                found.append({"item": self._display(a), "in_stock": q > 0, "qty": q})
                if q > 0 and a not in self.alternatives_seen:
                    self.alternatives_seen.append(a)
            return EnvResult(S.ToolResult("search_alternative",
                                          {"query": self._display(item), "alternatives": found}),
                             "alternatives")
        if action == "place_order":
            if not item:
                return self._error("missing argument: item")
            if stock.get(item, 0) <= 0:
                self.violations.append(f"ordered unavailable item {item!r}")
                return EnvResult(S.ToolResult("place_order", {"status": "failed", "error": "out_of_stock",
                                                              "item": self._display(item)}), "status")
            ok_target = item in self.orderable()
            if not ok_target:
                self.violations.append(f"ordered an item that was not requested: {item!r}")
            self.orders.append(item)
            for k in self.state["stock"]:
                if _norm(k) == item:
                    self.state["stock"][k] -= 1
            res = EnvResult(S.ToolResult("place_order", {"status": "confirmed",
                                                         "order_id": S._order_id(self.task.task_id, item),
                                                         "item": self._display(item)}), "status")
            if ok_target:
                self._finish("order confirmed", True)
                return None
            return res
        if action == "ask_user":
            self._finish("handed over to the user", bool(self.alternatives_seen) and not self.target_available())
            return None
        if action in ("stop_no_option", "proceed"):
            self._finish(f"stopped with {action}", False)
            return None
        return self._error(f"{action} is not applicable here")

    def _documents(self, action: str, args: dict[str, Any]) -> EnvResult | None:
        acl = self._acl()
        if action == "request_access":
            doc = _norm(args.get("doc_id")) or self.target
            if doc not in acl:
                return self._error(f"unknown document {args.get('doc_id')!r}")
            if self.access and doc == self.target:
                return EnvResult(S.ToolResult("request_access", {"doc_id": self._display_doc(),
                                                                 "status": "already_granted"}), "status")
            if doc == self.target:
                self.access = True
                content = {_norm(k): v for k, v in self.state.get("content", {}).items()}.get(doc, "")
                return EnvResult(S.ToolResult("request_access", {"doc_id": self._display_doc(),
                                                                 "status": "approved", "access": "granted",
                                                                 "content": content}), "access")
            return self._error("access can only be requested for the document in question")
        if action == "proceed":
            if not self.access:
                self.violations.append("proceeded without access to the document")
            self._finish("proceeded", self.access)
            return None
        if action in ("ask_user", "stop_no_option"):
            self._finish(f"stopped with {action}", False)
            return None
        return self._error(f"{action} is not applicable here")


# --------------------------------------------------------------------------
# Прогон эпизода
# --------------------------------------------------------------------------


@dataclass
class EpisodeResult:
    task_id: str
    branch: int
    pair_type: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    critical_ok: bool = False
    goal: bool = False
    violations: list[str] = field(default_factory=list)
    reason: str = ""

    @property
    def success(self) -> bool:
        return self.critical_ok and self.goal and not self.violations

    def to_dict(self) -> dict[str, Any]:
        return {"task_id": self.task_id, "branch": self.branch, "pair_type": self.pair_type,
                "success": self.success, "critical_ok": self.critical_ok, "goal": self.goal,
                "violations": self.violations, "reason": self.reason,
                "n_steps": len(self.steps), "steps": self.steps}


# decide(messages, essential_field, expected, step_index) -> (action, arguments, info)
Agent = Callable[[list[dict[str, Any]], "str | None", frozenset, int], tuple[str, dict, dict]]


def rollout(task: S.PairedTask, branch: int, messages: list[dict[str, Any]], agent: Agent,
            max_steps: int = 4) -> EpisodeResult:
    """Эпизод от точки решения пары.

    messages — контекст метода в точке решения (последнее сообщение — критический
    ответ инструмента). Продолжение дописывается в том же формате: вызов агента
    и правдивый ответ среды.
    """
    env = Env(task, branch)
    msgs = list(messages)
    essential = task.essential_field(branch)
    ep = EpisodeResult(task.task_id, branch, task.pair_type)
    for i in range(max_steps):
        expected = env.expected()
        action, args, info = agent(msgs, essential, expected, i)
        ok = action in expected
        if i == 0:
            ep.critical_ok = ok
        res = env.step(action, args)
        ep.steps.append({"i": i, "action": action, "arguments": args, "expected": sorted(expected),
                         "ok": ok, "observation": None if res is None else res.result.render(env.style),
                         **info})
        if env.done:
            break
        assert res is not None
        msgs = msgs + [S.tool_call_message(action, args), S.tool_message(res.result, env.style)]
        try:
            essential = S.essential_substring(res.result.render(env.style), res.essential_key)
        except ValueError:
            essential = None
    else:
        env.reason = env.reason or f"step budget ({max_steps}) exhausted"
    ep.goal, ep.violations, ep.reason = env.goal, list(env.violations), env.reason
    return ep
