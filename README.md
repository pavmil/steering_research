# Causal Steering of Tool-Feedback Integration in LLM Agents

Исследование: можно ли вмешательством в активации (steering) научить агента правильно
менять план по смыслу полученного результата инструмента — и выгоднее ли это, чем
промпт, явное ведение состояния и дообучение.

**Статус:** постановка зафиксирована, инфраструктура готова: все методы сравнения
(промпт, пересказ, явное состояние, steering, steering с гейтом, адаптация TACT, LoRA) и
многошаговые эпизоды. Первые результаты на Qwen3-1.7B (постановка v3, только val, test не
открыт): разрыв «различает, но не действует» есть; CAA-steering закрывает его на +3 п.п.,
LoRA на тех же данных — на +33 п.п. История и результаты — в [docs/JOURNAL.md](docs/JOURNAL.md).

## Ядро проекта в одном абзаце

Агент проверил наличие товара. Инструмент вернул «в наличии» → оформлять заказ;
«нет в наличии» → искать альтернативу. Берём одинаковую историю и один вызов инструмента,
меняем **только** правдивый ответ — правильные следующие действия в двух ветках разные.
Смотрим, меняет ли агент действие как надо, и помогает ли steering.

## Быстрый старт (Mac)

```bash
uv venv --python 3.12 .venv      # системный python3 на macOS — 3.9, проекту нужен ≥ 3.10
uv pip install --python .venv/bin/python -e ".[model,notebooks,dev,finetune]"
source .venv/bin/activate
stf-doctor                      # видит ли этот python torch/transformers и MPS
pytest                          # ~1–2 мин, без модели и сети
jupyter lab notebooks/
```

Дальше по порядку: `01_smoke_mock` → `02_model_check` → `03_h1_h2_baseline` → `04_h3_steering`.
Новый эксперимент = копия `00_TEMPLATE.ipynb` с другой ячейкой конфига. В конце
ноутбука вывод пишется в журнал. Как всё устроено — [docs/PIPELINE.md](docs/PIPELINE.md).

## Навигация

| Файл | Что внутри |
|---|---|
| [docs/JOURNAL.md](docs/JOURNAL.md) | **Журнал**: вся история — прогоны, решения, заметки |
| [docs/PIPELINE.md](docs/PIPELINE.md) | Как устроен пайплайн экспериментов, гарантии протокола, расширение |
| [docs/HYPOTHESES.md](docs/HYPOTHESES.md) | Зафиксированные гипотезы H1–H4 и критерии опровержения |
| [docs/METRICS.md](docs/METRICS.md) | Зафиксированные метрики с источниками |
| [docs/EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md) | Матрица методы × контексты, поэтапный план |
| [docs/RELATED_WORK.md](docs/RELATED_WORK.md) | Литература, проверка свободы ниши |
| [CONTEXT.md](CONTEXT.md) | Контекст проекта, что отвергли и почему |
| [AGENTS.md](AGENTS.md) | Правила работы в репозитории для AI-агентов |

## Структура

```
notebooks/                     эксперименты: 00_TEMPLATE + 01–04
configs/                       YAML-конфиги для запуска без ноутбука (stf-run --config)
docs/                          гипотезы, метрики, план, литература, PIPELINE, JOURNAL
src/steering_tool_feedback/    код (устанавливаемый пакет)
  config.py                    конфиг эксперимента
  pipeline.py                  стадии data → … → test, протокол, сводки
  backends/                    HF (CPU / MPS / CUDA) и mock
  methods.py                   baseline, strong_prompt, restate, explicit_state, steering,
                               steering_gated, tact, lora, sham-контроли
  gates.py  journal.py         вердикты H1–H4; журнал
  simulator.py  windows.py     парные миры; окна позиций
  episodes.py                  многошаговая среда от точки решения пары, verifier
  finetune.py                  LoRA/SFT на тех же данных (H4)
  steering.py  probe.py        хуки и векторы; probe
  metrics.py  actions.py       метрики; формат вызова и разбор действия
  store.py  viz.py             файлы и кэш; графики
tests/                         pytest (в т.ч. исполнение ноутбуков)
results/                       локально; в git — только index, test_lock, frozen, protocol
skills/                        внешние скиллы (см. AGENTS.md про доверие)
```

## Установка

Ставить **в тот Python, которым будешь запускать** — глобальный или venv, неважно,
главное один и тот же. Из корня репозитория:

```bash
python3 -m pip install -e ".[model,notebooks,dev]"   # всё
python3 -m pip install -e .                          # только ядро (mock, метрики, симулятор)
stf-doctor                                           # какой python запущен и что он видит
```

`-e` — editable: правки в `src/` подхватываются без переустановки, импорт работает
из любой папки (`from steering_tool_feedback import metrics`), без `sys.path`.
Чтобы Jupyter видел пакет, ядро ноутбука должно быть тем же Python — `stf-doctor`
в первой ячейке `02_model_check` это покажет.

Если `stf-*` «command not found» — скрипты встали в папку не из PATH (типично для
`pip install --user`: `~/Library/Python/3.x/bin` на macOS, `~/.local/bin` на Linux).
`stf-doctor` покажет, куда именно и что дописать в `~/.zshrc`. Без PATH работает всегда:

```bash
python3 -m steering_tool_feedback doctor
python3 -m steering_tool_feedback windows --tokenizer <...>
python3 -m steering_tool_feedback run --mock
python3 -m steering_tool_feedback note "текст"
```

## Команды

```bash
pytest                                          # все тесты, включая исполнение ноутбуков на mock
STF_TOKENIZER=<папка токенизатора> pytest       # + тесты на реальном chat template
stf-windows --tokenizer <папка или HF id>       # окна позиций глазами
stf-run --mock                                  # весь пайплайн на mock
stf-run --config configs/qwen3-1.7b-mac.yaml --until probe
stf-note "решили сократить alpha до 3 значений" --tag решение
```

Черновые прогоны, которые не должны попадать в журнал и `results/`:
`STF_RESULTS_DIR=/tmp/r STF_JOURNAL=/tmp/j.md stf-run --mock`.

## Команда

Павел (постановка), Иван (ревью).
