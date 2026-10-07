"""Seed-separated Russian memory tasks with task-independent filler."""
import random

CATEGORIES = ['semantic', 'exact', 'associative', 'overwrite', 'forget', 'temporal', 'multihop']
FILLERS = [
    'Вечером на улице стало прохладно, и прохожие надели куртки.',
    'В библиотеке открыли выставку старых карт и научных журналов.',
    'За окном шумели деревья, а над рекой медленно поднимался туман.',
    'Участники обсуждали устройство двигателей и историю морских путешествий.',
    'После обеда сотрудники вернулись к обычной работе в мастерской.',
    'На столе лежали газета, карандаш и открытая записная книжка.',
    'Рассказ продолжался, постепенно переходя к описанию соседнего города.',
    'Погода оставалась спокойной, хотя прогноз обещал сильный дождь.',
]
COLORS = ['красный', 'синий', 'зелёный', 'жёлтый', 'белый', 'чёрный', 'серый', 'оранжевый']


def task(rng, category, split='train'):
    names = ['Анна','Иван','Ольга','Павел','Елена','Олег'] if split=='train' else ['Мария','Борис','Вера','Денис','Нина','Юрий']
    name = rng.choice(names)
    code, old = str(rng.randrange(100, 1000)), str(rng.randrange(100, 1000))
    if category == 'semantic':
        answer = rng.choice(COLORS)
        fact = f'Цвет автомобиля, которым владеет {name}, — {answer}.'
        question = f'Какого цвета машина этого человека ({name})? Ответ:'
    elif category == 'exact':
        answer = ''.join(rng.choice('0123456789abcdef') for _ in range(12))
        fact = f'Секретный идентификатор для пользователя {name}: {answer}.'
        question = f'Повтори секретный идентификатор пользователя {name}. Ответ:'
    elif category == 'associative':
        answer = code; fact = f'Пользователь {name} хранит вещи в ячейке с номером {code}.'
        question = f'Какой номер ячейки у пользователя {name}? Ответ:'
    elif category == 'overwrite':
        answer = code; fact = f'Код устройства был {old}. Затем код устройства изменили на {code}.'
        question = 'Какой текущий код устройства? Ответ:'
    elif category == 'forget':
        answer = code; fact = f'Старый код {old} больше не актуален. Забудь его. Новый действующий код: {code}.'
        question = 'Какое значение кода актуально сейчас? Ответ:'
    elif category == 'temporal':
        events = rng.sample(['дождь','снег','ветер','туман','град','гроза'],3)
        answer = events[1]; fact = f'Наблюдения по порядку: сначала {events[0]}, затем {events[1]}, наконец {events[2]}.'
        question = 'Какое явление наблюдали вторым? Ответ:'
    elif category == 'multihop':
        answer = code; box = rng.randrange(1, 20); key = rng.choice(['А','Б','В','Г','Д'])
        fact = f'{name} владеет коробкой {box}. В коробке {box} находится ключ {key}. Ключ {key} открывает комнату {code}.'
        question = f'Какую комнату может открыть {name}? Ответ:'
    else: raise ValueError(category)
    # Held-out surface form in addition to disjoint seeds and person names.
    if split != 'train':
        fact = 'В архиве зафиксировано следующее. ' + fact
        question = 'Используя запись из архива, ответь. ' + question
    return fact, question, answer


def token_task(sp, rng, category, distance, split='train'):
    fact, question, answer = task(rng, category, split)
    # Put obsolete and replacement facts in different chunks. The final gap
    # still measures distance from the latest relevant fact to the question.
    update_gap=0
    if category in ['overwrite','forget']:
        marker='Затем' if category=='overwrite' else 'Новый действующий'
        old,new=fact.split(marker,1)
        old_ids=sp.encode(old); new_ids=sp.encode(marker+new)
        update_gap=512
        between=[]
        while len(between)<update_gap: between.extend(sp.encode(rng.choice(FILLERS)))
        prefix=[sp.bos_id()]+old_ids+between[:update_gap]+new_ids
    else:
        prefix = [sp.bos_id()] + sp.encode(fact)
    # Distance is exactly the number of token positions from the end of the
    # fact to the beginning of the question, not characters or total length.
    filler = []
    while len(filler) < distance:
        filler.extend(sp.encode(rng.choice(FILLERS)))
    prompt = prefix + filler[:distance] + sp.encode(question)
    answer_ids = sp.encode(' ' + answer) + [sp.eos_id()]
    return prompt, answer_ids, {'category':category, 'distance':distance,
                               'fact_end':len(prefix), 'question_start':len(prefix)+distance,
                               'answer':answer, 'fact':fact, 'question':question,'update_gap':update_gap}
