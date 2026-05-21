# MAGNIT HSE Uplift Modeling

Короткий baseline для кейса по uplift-моделированию: нужно ранжировать клиентов по ожидаемому приросту `rec_spend` от персональной коммуникации.

## Что внутри

- `hurdle_t_learner_baseline.ipynb` — основной ноутбук с EDA, валидацией и инференсом.
- `requirements.txt` — зависимости для запуска.
- `.gitignore` — чтобы в репозиторий не улетели данные, IDE-файлы, модели и локальные черновики.

## Идея решения

Обычная модель на `rec_spend` отвечает на вопрос “кто и так много потратит”. Для этой задачи важнее другой вопрос: “у кого траты вырастут именно из-за промо”.

Поэтому baseline построен как `Hurdle T-learner` на CatBoost:

1. отдельно учим модели для `treatment` и `control`;
2. отдельно оцениваем вероятность ненулевой покупки;
3. отдельно оцениваем сумму покупки при условии, что покупка была;
4. считаем uplift как разницу ожидаемых трат:

```text
uplift = E[rec_spend | treatment=1] - E[rec_spend | treatment=0]
```

Такой подход лучше подходит под данные, потому что около 90% значений `rec_spend` равны нулю.

## Валидация

В ноутбуке есть:

- `StratifiedShuffleSplit` по связке `treatment_flg + communication_type`;
- `StratifiedKFold` по той же связке;
- метрика `uplift@10`;
- bootstrap lower bound для top-10%;
- stress-test по верхним `user_id`, потому что test идет следующим диапазоном id.

Это не time series validation: явной даты в данных нет, поэтому делить по времени здесь было бы искусственно.

## Как запустить

1. Положить данные локально в папку `кей/`:

```text
кей/train.parquet
кей/test.parquet
кей/data_description.xlsx
```

2. Установить зависимости:

```bash
pip install -r requirements.txt
```

3. Открыть и запустить:

```text
hurdle_t_learner_baseline.ipynb
```

После финальной ячейки появится `predictions.csv` с колонками:

```text
user_id,UPLIFT_SCORE
```

Файл `predictions.csv` не коммитится, потому что это генерируемый результат.
