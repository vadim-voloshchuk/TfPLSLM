"""Regenerate the final report from saved metrics and reviewed interpretation."""
import json
import math
from pathlib import Path
import statistics
import time
import sys
make_plots='--no-plots' not in sys.argv
if make_plots:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

root=Path('.'); reports=root/'reports';reports.mkdir(exist_ok=True)
plots=reports/'plots';plots.mkdir(exist_ok=True)
def read(path,default=None):
    p=Path(path);return json.loads(p.read_text()) if p.exists() else default
def metrics(name):
    p=root/f'artifacts/{name}/metrics.jsonl'
    return [json.loads(s) for s in p.read_text().splitlines()] if p.exists() else []

names=[n for n in ['poc','ssm','transformer'] if Path(f'artifacts/{n}/final_training.json').exists()]
manifest=read('data/manifest.json');bench=read('artifacts/benchmark64/benchmark.json')
budget=read('artifacts/environment/budget.json');smoke=read('artifacts/smoke/smoke.json')
pilot=read('artifacts/short_real/final_training.json')
pilot_initial=read('artifacts/short_real/initial_validation.json')
final={n:read(f'artifacts/{n}/final_training.json') for n in names}
validation={n:read(f'artifacts/{n}/eval/validation.json',final[n]) for n in names}
memory={n:read(f'artifacts/{n}/eval/memory_results.json',[]) for n in names}
interpretation=read('reports/interpretation.json')
if not interpretation:
    raise RuntimeError('Write a reviewed reports/interpretation.json before publishing the final report')
if make_plots:
    plt.style.use('seaborn-v0_8-whitegrid')
    fig,ax=plt.subplots(figsize=(8,4.2))
    for name in names:
        m=[r for r in metrics(name) if 'loss' in r]
        ax.plot([r['tokens_seen']/1e6 for r in m],[r['loss'] for r in m],label=name,linewidth=1)
    ax.set(xlabel='Supervised tokens (millions)',ylabel='Training cross-entropy (nats)',title='Training on identical data streams')
    ax.legend();fig.tight_layout();fig.savefig(plots/'training_loss.png',dpi=160);plt.close(fig)

scale=read('artifacts/poc/eval/memory_scaling.json',[])
if scale and make_plots:
    fig,ax=plt.subplots(figsize=(7,4))
    ax.plot([r['history_tokens'] for r in scale],[r['state_bytes']/2**20 for r in scale],'o-',label='Persistent state')
    ax.set(xscale='log',xlabel='Processed history tokens',ylabel='MiB per example',title='Persistent state storage, not total training memory')
    ax.set_ylim(0,max(r['state_bytes']/2**20 for r in scale)*1.3);ax.legend();fig.tight_layout()
    fig.savefig(plots/'state_memory.png',dpi=160);plt.close(fig)

rows=[]
rows += ['# Итоговый отчёт TfPLSLM','',
         *(['> Промежуточная версия: POC завершён; независимый SSM-контроль ещё обучается. Его результаты и окончательный статус аренды будут добавлены.',''] if 'ssm' not in names else []),
         interpretation['verdict'],
         '', '## Реализация и архитектура','',
         f"Основная модель: **{bench['parameters']:,} параметров**. 16 блоков Mamba-2-style SSD, d_model=512, d_state=64, expand=2, head_dim=64, FFN=1536; связанная входная/выходная матрица эмбеддингов.",
         'Triton SSD scan взят из [state-spaces/mamba](https://github.com/state-spaces/mamba/tree/95d8aba8a8c75aedcaa6143713b11e745e7cd0d9), commit `95d8aba8a8c75aedcaa6143713b11e745e7cd0d9`. Основной путь не содержит self-attention или KV-cache. Это собственная обвязка SSD + FFN, а не полная копия архитектуры Mamba-2.',
         'Явная память — 16×256 float32. Она считывается до чанка и обновляется после 512 токенов. Два последовательных чанка имеют общий граф градиентов (1024 токена), после чего состояние отсоединяется; перенос состояния продолжается внутри документа. Граница документа сбрасывает всю строку состояния.',
         'Каждый из 16 слоёв хранит SSD state 16×64×64 float32 и 3 предыдущих свёрточных входа. Дополнительно хранятся 4096 чисел явной памяти и сумма скрытых векторов незавершённого чанка. SSM-only содержит **72 609 024** параметра; Transformer — **75 883 520**, 16 слоёв, 8 attention heads, FFN=3072, окно 512 токенов.',
         '', '## Данные и токенизатор','',
         f"Источник: [Cultura-Ru-Edu](https://huggingface.co/datasets/deepvk/cultura_ru_edu), revision `{manifest['revision']}`.", '',
         '| Раздел | Документы | Подготовленные токены |','|---|---:|---:|']
for key in ['train','synthetic','validation']:
    d=manifest[key];rows.append(f"| {key} | {d['documents']:,} | {d['tokens']:,} |")
rows += ['',f"SentencePiece BPE, vocab=16 384, byte fallback, нормализация identity. Обучающая выборка tokenizer: {manifest['tokenizer_sample']['documents']:,} документов. SHA-256: `{manifest['tokenizer_sha256']}`.",
         'Train/validation имеют нулевое пересечение точных текстовых SHA-256; near-duplicate удаление не выполнялось. Документы обучения выбираются с возвращением: число просмотренных токенов не равно числу уникальных токенов.',
         'Синтетика использует отдельный test seed, новые имена и изменённые формулировки. Overwrite/forget разделяют старый и новый факт 512 токенами. Дистанция benchmark — точный токенный промежуток от конца последнего факта до начала вопроса.',
         'Обучающие дистанции синтетики: 64, 128, 256, 512, 1024 и 2048 токенов, с повышенной долей 512. Проверки 8192 и 32768 измеряют перенос за пределы обучающего распределения дистанций.',
         '', '## Проверки и производительность','',
         f"Tiny overfit в уменьшенной конфигурации `configs/smoke.json`: loss {smoke['first_loss']:.6f} → {smoke['final_loss']:.6f}. Разница параметров после сохранения/восстановления следующего optimizer step: {smoke['resume_max_parameter_difference']:.1e}.",
         f"Отдельный real-data pilot полной модели: {pilot['tokens_seen']:,} токенов, validation loss {pilot_initial['validation_loss']:.4f} → {pilot['validation_loss']:.4f} (входная проверка на 8 batches, финальная на 32). До основного обучения прошли preflight генерации, memory evaluation и оценки размера состояния.",
         'Пять основных unit tests прошли: причинность, эквивалентность полного чанка и пошагового инференса, градиенты записи памяти и сброс строк, Triton/reference forward+backward, document boundaries и sampler resume. Отдельный шестой тест проверяет причинность и backward Transformer. Основной прогон также был намеренно остановлен и успешно восстановлен на шаге 264 (12 428 934 токена), включая optimizer и 24 активные строки документов.',
         f"H100 PCIe 80 GB; PyTorch 2.6.0+cu124. Batch=64, chunk=512, unroll=2. После прогрева: **{bench['tokens_per_sec']:,.0f} tokens/s**, {bench['stable_steps']} стабильных шагов, peak allocated **{bench['peak_allocated_gib']:.2f} GiB**, GPU utilization {bench['gpu_utilization']:.0f}%.",
         'Benchmark использовал заранее размещённые случайные токены. Ниже приведена фактическая скорость с padding и документным загрузчиком.',
         '', '## Обучение и language modelling','',
         '| Модель | Tokens seen | Natural / synthetic | Шаги | Время, с | Median useful tok/s | Validation loss | PPL |',
         '|---|---:|---:|---:|---:|---:|---:|---:|']
for name in names:
    f=final[name];v=validation[name];rates=[r['tokens_per_sec'] for r in metrics(name) if 'tokens_per_sec' in r and r.get('step',0)>100]
    rows.append(f"| {name} | {f['tokens_seen']:,} | {f['source_tokens'][0]:,} / {f['source_tokens'][1]:,} | {f['step']:,} | {f['seconds']:.1f} | {statistics.median(rates):,.0f} | {v['validation_loss']:.5f} | {v['perplexity']:.2f} |")
rows += ['', 'AdamW (betas 0.9/0.95), LR 6e-4, cosine decay до 10%, warmup 100 шагов, clip 1.0, bf16 autocast. Оба SSM-прогона используют одинаковый sampler seed и токенный бюджет.',
         ('Transformer также обучен на том же потоке данных и токенном бюджете. Его контекст ограничен 512 токенами, без переноса через чанки; это контроль языкового моделирования, а не равный по доступной истории long-context конкурент.' if 'transformer' in names else 'Transformer baseline подготовлен в коде и конфигурации; результаты его полного обучения здесь не заявляются.'),
         'Final validation использует 64 одинаковых документных batches с тем же seed у всех моделей. Промежуточные проверки использовали 8 batches; непосредственно final_training — 32. Сравнивать промежуточные и финальные PPL как один и тот же набор нельзя. Training loss включает синтетику и напрямую с natural-only validation loss не сопоставим.',
         '', '| Модель | Peak train GiB | Median GPU % | Median loader wait / logged interval, с | Последний train loss |', '|---|---:|---:|---:|---:|']
for name in names:
    m=[r for r in metrics(name) if 'loss' in r]
    rows.append(f"| {name} | {max(r['peak_allocated_gib'] for r in m):.2f} | {statistics.median(r['gpu_utilization'] for r in m):.0f} | {statistics.median(r['dataloader_wait_seconds'] for r in m):.3f} | {m[-1]['loss']:.5f} |")
rows += ['', interpretation['lm_quality'],
         '', '![Training loss](plots/training_loss.png)', '', '## Memory benchmark: normal state','',
         '| Distance | Semantic | Exact | Associative | Overwrite | Forget | Temporal | Multi-hop |',
         '|---:|---:|---:|---:|---:|---:|---:|---:|']
cats=['semantic','exact','associative','overwrite','forget','temporal','multihop']
for dist in [512,2048,8192,32768]:
    vals=[]
    for cat in cats:
        r=next((r for r in memory.get('poc',[]) if r['mode']=='normal' and r['distance']==dist and r['category']==cat),None)
        vals.append(f"{r['correct']}/{r['n']} ({100*r['accuracy']:.1f}%)" if r else 'не измерено')
    rows.append('| '+str(dist)+' | '+' | '.join(vals)+' |')
rows += ['', 'Метрика — exact match сгенерированного ответа целиком (допускается конечная точка). Answer NLL — отдельная teacher-forced диагностика, она не заменяет accuracy. JSON содержит интервалы Wilson 95%, отдельные ответы и donor labels для shuffled state.',
         '', '## Memory ablations: средняя accuracy по семи категориям','',
         '| Distance | Normal | Zero | Reset | Shuffle | No memory | SSM-only |',
         '|---:|---:|---:|---:|---:|---:|---:|']
for dist in [512,2048,8192,32768]:
    values=[]
    for name,mode in [('poc','normal'),('poc','zero'),('poc','reset'),('poc','shuffle'),('poc','no_memory'),('ssm','normal')]:
        selected=[r for r in memory.get(name,[]) if r['distance']==dist and r['mode']==mode]
        values.append(f"{100*sum(r['correct'] for r in selected)/sum(r['n'] for r in selected):.2f}%" if selected else 'не измерено')
    rows.append('| '+str(dist)+' | '+' | '.join(values)+' |')
rows += ['', 'Категории имеют одинаковое число примеров, но разные пространства ответов; это описательное среднее, не общий chance-adjusted score.',
         'Ориентир случайного угадывания правильной метки при известной категории: 12,5% для цвета, 16,7% для temporal, 1/900 для трёхзначных кодов и 1/16¹² для exact ID. Их равновзвешенное среднее — около 4,23%; это теоретический ориентир, а не измеренная генеративная baseline.',
         '32 примера на категорию/дистанцию; не 32 независимых обучающих запуска. При 0/32 верхняя граница Wilson 95% около 10,7%, поэтому нулевое наблюдение не доказывает абсолютно нулевую способность.',
         '', interpretation['memory_quality'],
         '', '## Ablations на обычном русском тексте','', '| Модель | State | Loss | Δ к normal |','|---|---|---:|---:|']
for name in names:
    for r in read(f'artifacts/{name}/eval/lm_ablations.json',[]):
        rows.append(f"| {name} | {r['mode']} | {r['loss']:.6f} | {r['loss_delta_from_normal']:+.6f} |")
rows += ['', 'Zero обнуляет состояние перед каждым чанком; reset — перед каждой парой чанков; shuffle подставляет другую строку; no_memory отключает явную память, сохраняя SSD state. На synthetic benchmark интервенция делается перед одинаковым завершающим чанком; reset оставляет только последний полный чанк истории.',
         'Ablation no_memory изменяет уже обученную модель, а SSM-only обучен независимо. Они отвечают на разные вопросы. Одинаковый seed не означает идентичную инициализацию общих весов из-за различного порядка создания модулей.',
         '', '## Контрфактические проверки','', '| Модель | Подмена | Выход для красного / синего префикса | Mean abs logit Δ | KL |', '|---|---|---|---:|---:|']
for name in [n for n in names if n!='transformer']:
    for r in read(f'artifacts/{name}/eval/counterfactual.json',[]):
        outputs=' / '.join(s.replace('\n',' ').replace('|','\\|') for s in r['outputs'])
        rows.append(f"| {name} | {r['mode']} | {outputs} | {r.get('mean_absolute_logit_change',0):.6g} | {r.get('kl_from_normal',0):.6g} |")
rows += ['', interpretation['causal_state'],
         '', '| Дистанция | Категория | Accuracy исходного ответа при shuffle | Accuracy ответа донора |', '|---:|---|---:|---:|']
for r in memory.get('poc',[]):
    if r['mode']=='shuffle' and r['distance'] in [512,2048]:
        rows.append(f"| {r['distance']} | {r['category']} | {100*r['accuracy']:.2f}% | {100*r['donor_answer_accuracy']:.2f}% |")
rows += ['', 'Donor accuracy — совпадение с меткой того примера, чьё состояние было подставлено. В этих задачах оно информативнее простого изменения logits. Совпадения меток между разными примерами возможны, поэтому для малых пространств ответов их нужно сопоставлять с normal/zero.',
         '', '## Диагностика состояния','', '| Модель | Последний SSD RMS | Max abs за прогон | Последний memory RMS | Retain mean | Write mean | Dead slot fraction |', '|---|---:|---:|---:|---:|---:|---:|']
for name in [n for n in names if n!='transformer']:
    m=[r for r in metrics(name) if 'ssm_rms' in r];r=m[-1]
    rows.append(f"| {name} | {r['ssm_rms']:.6g} | {max(x['ssm_max_abs'] for x in m):.6g} | {r['memory_rms']:.6g} | {r.get('retain_mean',0):.6g} | {r.get('write_mean',0):.6g} | {r.get('dead_slot_fraction',0):.3f} |")
rows += ['', 'Для SSM-only нули в столбцах явной памяти означают отсутствие модуля. Метрики gates — среднее по последнему логированному batch, dead определяется RMS < 1e-5. Это не мера полезности слотов; причинные ablations важнее амплитуды состояния. Полные gate saturation и slot RMS сохранены в metrics.jsonl.',
         'Отдельная диагностика на natural validation исключает padding и переходы через границы документов: 86,97% изменений слотов имеют RMS < 1e-5; медиана retain=0,99950 и write=0,00240. Один слот существенно активнее остальных. На обычном тексте writer почти закрыт; это согласуется с малым LM-эффектом no_memory, но не означает, что память не работает на синтетике. Средняя абсолютная корреляция координат слотов 0,016 — это диагностическая статистика, не доказательство независимости представлений.',
         '', '## Рабочая память','', '| История | State bytes | Allocated bytes | Peak allocated bytes |','|---:|---:|---:|---:|']
for r in scale: rows.append(f"| {r['history_tokens']:,} | {r['state_bytes']:,} | {r['allocated_bytes']:,} | {r['peak_allocated_bytes']:,} |")
rows += ['', '![State memory](plots/state_memory.png)',
         'Это фиксированный размер recurrent working state при фиксированном batch/chunk, а не утверждение O(1) о всей системе или обучении.',
         '', '## Примеры генерации','']
for r in read('artifacts/poc/eval/generation_samples.json',[]):
    rows += [f"**Prompt:** {r['prompt']}",'',r['continuation'],'']
rows += ['## Ограничения и научная интерпретация','',
         'Один seed на вариант; малый токенный бюджет для модели 77M; ограниченная выборка benchmark; фиксированные шаблоны и filler; градиентный горизонт 1024 токена; явная память обновляется усреднённым представлением чанка. По одному такому прогону нельзя делать вывод о превосходстве над Transformer или об общей непригодности bounded memory.',
         'Синтетика состоит из семи простых семейств и восьми повторяемых filler-предложений. Отдельные слова ответов temporal могут случайно встретиться в filler независимо от правильной метки. Длина и шаблоны отличаются от естественных длинных документов; это ограниченный диагностический benchmark.',
         'Multi-hop содержит одну цепочку без конкурирующих связей; успех можно получить, удержав её конечный номер, без общего композиционного рассуждения. Overwrite/forget проверяют актуальное значение в простой последовательности, а не произвольное адресное удаление множества фактов. Одновременная ёмкость для большого набора независимых фактов не измерена.',
         '', interpretation['failures'],
         '', 'В подготовленной синтетике ответы вместе с EOS занимают **346 452 из 49 937 710** supervised токенов (0,694%). При смеси 1/7 это около **0,099% позиций, по которым усредняется общий LM loss**. Поэтому снижение общего loss само по себе почти ничего не говорит об обучении recall. Это измеренная доля токенов, не доказательство причины провала. Метод подсчёта сохранён в `scripts/analyze_synthetic_supervision.py`.',
         '', '## Артефакты и воспроизводимость','',
         '`artifacts/<model>`: config, metrics, полный last.pt, eval. `data`: tokenizer, uint16 shards, индексы границ, manifest и хэши. `artifacts/environment`: версии библиотек, аппаратное окружение и ставка аренды. Полный checkpoint содержит optimizer, RNG, scheduler position, sampler cursors и recurrent state. Код и небольшие результаты опубликованы в GitHub; большие checkpoints и данные сохранены локально, вне Git.',
         '', '## Стоимость','',
         f"Зафиксированная ставка аренды: ${budget['dph_total']:.6f}/час. Итоговая оценка и статус остановки фиксируются отдельно после копирования результатов. Это расчёт по времени, не выписка Vast.ai.",
         '', '## Вывод и следующий эксперимент','',
         interpretation['conclusion'], '', interpretation['next_experiment']]
runtime=read('reports/runtime_summary.json')
if runtime:
    rows += ['', '## Завершение аренды и локальные копии','',
             runtime['status_text'], '',
             f"Время аренды до команды остановки: **{runtime['rental_hours']:.3f} ч**, оценка по зафиксированной total-ставке: **${runtime['estimated_running_cost']:.2f}**. В эту оценку не включены отдельные bandwidth charges и хранение после остановки.",
             'Контрольные суммы checkpoints находятся в `checkpoint_verification.json`; данные и токенизатор сверены с manifest. Остановка сохраняет диск контейнера, поэтому storage продолжает тарифицироваться.']
(reports/'results').mkdir(exist_ok=True)
for name in names:
    summary={'training':final[name],'validation':validation[name],
             'memory':memory[name],'lm_ablations':read(f'artifacts/{name}/eval/lm_ablations.json',[]),
             'counterfactual':read(f'artifacts/{name}/eval/counterfactual.json',[])}
    (reports/'results'/f'{name}.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
(reports/'FINAL_REPORT.md').write_text('\n'.join(rows),encoding='utf-8')
print('REPORT_WRITTEN',flush=True)
