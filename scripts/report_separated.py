"""Audit and document the disjoint-role and E5 capacity experiment."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score,log_loss
from catboost import CatBoostClassifier
from scripts.separated_models import OUT,NAMES,VARIANTS,choose_variant,PRIORITY,HISTORY,bootstrap_difference
from scripts.semantic_embeddings import save_json,fingerprint
from train_maria import metrics

LABELS = dict(old_best='Прежнее лучшее усреднение',base64_s0_bootstrap_fine='64 компоненты: один seed',
    fields64_s0_bootstrap_fine='64 + цитата и документ',e5_128_s0_bootstrap_fine='128 компонентов: один seed',
    e5_256_s0_bootstrap_fine='256 компонентов: один seed',baseline_mean_bootstrap_fine='64 компоненты: три seed',
    selected_mean_bootstrap_fine='Внутренний выбор: три seed')


def audit(result):
    protocol = json.loads((OUT/'protocol.json').read_text())
    assert result['complete'] and result['config']==protocol
    quality,artifacts = {},{}
    for number in (1,2,3):
        folder = OUT/f'fold_{number}'
        report = json.loads((folder/'report.json').read_text())
        assert report['complete'] and report['config']==protocol and set(report['models'])==set(NAMES)
        roles = report['author_plan']
        assert all(not set(roles[a])&set(roles[b]) for a in roles for b in roles if a!=b)
        old = json.loads((ROOT/'results'/'maria'/f'fold_{number}.json').read_text())
        assert set(roles['fitting'])==set(old['fitting_authors'])
        assert set(roles['validation'])==set(old['validation_authors'])
        assert set(roles['stopping'])|set(roles['decision'])==set(old['calibration_authors'])
        keys = {role:pd.read_csv(folder/f'{role}_keys.csv') for role in roles}
        for role in roles:
            assert set(keys[role].author)==set(roles[role])
        validation = pd.read_csv(folder/'predictions.csv')
        assert validation[['author','post','label']].equals(keys['validation'])
        priority_report = json.loads((PRIORITY/f'fold_{number}'/'report.json').read_text())
        assert set(priority_report['author_plan']['fitting_authors'])==set(roles['fitting'])
        for new,source in [('semantic_pca64.joblib',HISTORY/f'fold_{number}'/'semantic_pca.joblib'),
                           ('visual_pca.joblib',HISTORY/f'fold_{number}'/'visual_pca.joblib')]+[
                           (f'{field}_pca.joblib',PRIORITY/f'fold_{number}'/f'{field}_pca.joblib') for field in ('quote_text','document_text')]:
            current,original = joblib.load(folder/new),joblib.load(source)
            for attribute in ('components_','mean_','explained_variance_'):
                np.testing.assert_array_equal(getattr(current,attribute),getattr(original,attribute))
        for name,expected in report['pca']['hashes'].items():
            assert fingerprint(folder/name)==expected
        for name,entry in report['fits'].items():
            assert entry['roundtrip_passed'] and entry['script_sha256']==fingerprint(ROOT/'scripts'/'separated_models.py')
            schema = json.loads((folder/f'{name}_schema.json').read_text())
            assert schema['stopping_authors']==roles['stopping'] and schema['decision_authors']==roles['decision']
            assert not set(schema['columns'])&{'author','post','label','reactions','resid','split'}
        fold_quality = {}
        for name in NAMES:
            prefix = name.removesuffix('_bootstrap_fine').removesuffix('_grid')
            probabilities = np.load(folder/f'{prefix}_decision_probabilities.npz')
            for role in ('decision','validation'):
                p = probabilities[role]
                assert p.shape==(len(keys[role]),3) and np.isfinite(p).all() and (p>=0).all()
                np.testing.assert_allclose(p.sum(axis=1),1,atol=1e-6)
            chosen = report['models'][name]['decision']
            pred = (probabilities['validation']*chosen['multipliers']).argmax(axis=1)
            np.testing.assert_array_equal(pred,validation[name])
            assert metrics(validation.label,pred)==report['models'][name]['metrics']
            inner = (probabilities['decision']*chosen['multipliers']).argmax(axis=1)
            assert abs(f1_score(keys['decision'].label,inner,average='macro')-chosen['calibration_macro_f1'])<1e-10
            if name.endswith('_bootstrap_fine'):
                p = probabilities['validation']
                fold_quality[prefix] = dict(log_loss=log_loss(validation.label,p,labels=[0,1,2]),
                    raw_argmax_macro_f1=f1_score(validation.label,p.argmax(axis=1),average='macro'))
        scores = {variant:report['models'][f'{variant}_s0_bootstrap_fine']['decision']['calibration_macro_f1'] for variant in VARIANTS}
        assert scores==report['screening_scores'] and choose_variant(scores)==report['selected_variant']
        for prefix,variant in [('baseline', 'base64'),('selected',report['selected_variant'])]:
            saved = np.load(folder/f'{prefix}_mean_decision_probabilities.npz')
            for role in ('decision','validation'):
                arrays = [np.load(folder/f'{variant}_s{i}_probabilities.npz')[role] for i in range(3)]
                np.testing.assert_allclose(saved[role],np.mean(arrays,axis=0),atol=1e-6)
        quality[str(number)] = fold_quality
        artifacts[str(number)] = {path.name:fingerprint(path) for path in folder.iterdir() if path.suffix in ('.cbm','.joblib','.npz','.json','.csv')}
    save_json(OUT/'artifact_audit.json',dict(passed=True,forecast_count=36,artifacts=artifacts,probability_quality=quality))


def plot(result):
    names = list(LABELS)
    fig,axes = plt.subplots(1,2,figsize=(13,5.5),gridspec_kw={'width_ratios':[1.5,1]})
    values = [result['oof'][name]['macro_f1'] for name in names]
    y = np.arange(len(names))
    axes[0].scatter(values,y,s=65,c=['#999999']+['#517cb2']*4+['#2f8067','#cb8037'],zorder=3)
    axes[0].grid(axis='x',alpha=.2)
    axes[0].set_yticks(y,list(LABELS.values()),fontsize=9)
    axes[0].invert_yaxis()
    axes[0].set_xlim(min(values)-.008,max(values)+.004)
    for i,value in enumerate(values):
        axes[0].text(value+.0002,i,f'{value:.4f}',va='center',fontsize=9)
    axes[0].set_xlabel('OOF macro-F1, отложенные авторы')
    axes[0].set_title('Сравнение моделей')
    for fold,entry in result['pca'].items():
        axes[1].plot([64,128,256],[100*entry['explained_variance'][str(n)] for n in (64,128,256)],'o-',label=f'Fold {fold}')
    axes[1].set_xticks([64,128,256])
    axes[1].set_ylabel('Сохранённая дисперсия E5, %')
    axes[1].set_xlabel('Размерность вложенного представления')
    axes[1].set_title('Больше информации ≠ гарантированный прирост')
    axes[1].legend()
    fig.suptitle('Раздельная остановка и настройка решения • только официальный train',fontsize=12)
    fig.tight_layout()
    for extension in ('png','svg'):
        fig.savefig(OUT/f'separated_results.{extension}',dpi=180,bbox_inches='tight')
    plt.close(fig)


def diagnostics():
    predictions = pd.read_csv(OUT/'oof_predictions.csv')
    fields = joblib.load(PRIORITY/'features.joblib')
    joined = predictions.merge(fields[['author','post','quote_text_available','document_text_available']],on=['author','post'],validate='one_to_one')
    slices = {}
    for field in ('quote_text','document_text'):
        part = joined.loc[joined[field+'_available'].eq(1)]
        slices[field] = dict(posts=len(part),authors=int(part.author.nunique()),class_counts=part.label.value_counts().sort_index().to_dict(),
            macro_f1={name:metrics(part.label,part[name])['macro_f1'] for name in LABELS},
            fields_minus_base_95=bootstrap_difference(part.label,part.fields64_s0_bootstrap_fine,part.base64_s0_bootstrap_fine,part.author))
    importance = {}
    probabilities = {variant:[] for variant in VARIANTS}
    labels = []
    for fold in (1,2,3):
        folder = OUT/f'fold_{fold}'
        labels.append(pd.read_csv(folder/'validation_keys.csv').label.to_numpy())
        by_variant = {}
        for variant in VARIANTS:
            probabilities[variant].append(np.load(folder/f'{variant}_s0_probabilities.npz')['validation'])
            heads = []
            for threshold in (0,1):
                model = CatBoostClassifier().load_model(str(folder/f'{variant}_s0_gt{threshold}.cbm'))
                values = model.get_feature_importance()
                extra_semantic = [i for i,name in enumerate(model.feature_names_) if name.startswith('semantic_pca_') and int(name.rsplit('_',1)[1])>=64]
                separate_fields = [i for i,name in enumerate(model.feature_names_) if name.startswith(('quote_text_','document_text_'))]
                heads.append(dict(extra_semantic=float(values[extra_semantic].sum()),separate_fields=float(values[separate_fields].sum()),trees=model.tree_count_))
            by_variant[variant] = heads
        importance[str(fold)] = by_variant
    y = np.concatenate(labels)
    quality = {}
    for variant,arrays in probabilities.items():
        p = np.concatenate(arrays)
        quality[variant] = dict(log_loss=log_loss(y,p,labels=[0,1,2]),brier=float(np.square(p-np.eye(3)[y]).sum(axis=1).mean()),raw_argmax_macro_f1=f1_score(y,p.argmax(axis=1),average='macro'))
    result = dict(slices=slices,feature_importance=importance,probability_quality=quality,
        caveat='PredictionValuesChange importance is descriptive, not causal or a significance test; field slices are exploratory')
    save_json(OUT/'diagnostics.json',result)
    return result


def write_report(result,diagnostic):
    def value(name):
        return f"{result['oof'][name]['macro_f1']:.4f}".replace('.',',')
    lines = ['## Раздельная остановка, цитата/документ и ёмкость E5 — 26 октября 2026','',
        f"Завершены три внешних разбиения; обучены **{result['classifiers_trained']} новых бинарных CatBoost**. Порядковая модель использует два порога y>0 и y>1 с согласованием вероятностей. Официальный test не использовался.",'',
        '### Четыре роли авторов','',
        'Прежние fitting-авторы и внешние validation-авторы сохранены. Прежняя внутренняя calibration-группа случайно разделена по авторам: половина только для ранней остановки, половина для множителей классов и выбора варианта. Ни один автор не совмещает роли. PCA обучены только на fitting-авторах. Параметры CatBoost сохранены: до 800 деревьев, глубина 6, learning rate 0,045, L2=5, patience=100.','',
        '| Fold | Обучение | Остановка | Решение | Внешняя оценка |','|---|---:|---:|---:|---:|']
    for fold,counts in result['author_counts'].items():
        lines.append('| '+fold+' | '+' | '.join(str(counts[key]) for key in ('fitting','stopping','decision','validation'))+' |')
    lines += ['', 'Сравнение с прежним протоколом меняет и разделение ролей, и размер групп для остановки/решения. Оно не измеряет изолированно пользу независимости данных. Прежняя внешняя CV тоже была по отложенным авторам; новая схема устраняет повторное использование именно внутренней группы. Метрика decision-группы после подбора не считается независимой оценкой.','',
        '### Три содержательных проверки','',
        'Контроль — порядковая модель с историей того же режима цитирования и прежними 64 E5-компонентами. В fields64 к ней добавлены раздельные представления цитаты и документа: PCA16 каждого поля, наличие, логарифм длины и сходство с собственным текстом. Отсутствующим полям присвоен нулевой вектор. Вспомогательные поля не добавлялись.','',
        'Варианты 128/256 сохраняют исходные 64 компоненты дословно, а дополнительные 64/192 направления получены PCA остатка реконструкции. Это вложенные расширения, а не заново подобранные независимые PCA128/PCA256. Основания проверены на ортогональность. Замороженный E5 не дообучался; заново обучались классификаторы.','',
        '| Fold | Дисперсия: 64 | 128 | 256 |','|---|---:|---:|---:|']
    for fold,entry in result['pca'].items():
        lines.append('| '+fold+' | '+' | '.join(f"{100*entry['explained_variance'][str(n)]:.1f}%" for n in (64,128,256))+' |')
    lines += ['', 'Все четыре варианта сначала обучены с одним фиксированным seed. Выбор для дальнейшего усреднения сделан только по macro-F1 decision-авторов с bootstrap-правилом; равенство разрешается в пользу более раннего варианта. Затем контроль и выбранный вариант обучены ещё на двух seed. Вероятности усреднены с равными весами. Поэтому прочие варианты в таблице — первичный одно-seed скрининг, а не полноценное сравнение трёх seed.','',
        'Внутренний выбор: '+', '.join(f'fold {fold}: {variant}' for fold,variant in result['selected_variants'].items())+'.','',
        '| Вариант | Macro-F1 | Без своего текста | С цитатой | История < 10 |','|---|---:|---:|---:|---:|']
    for name,label in LABELS.items():
        lines.append('| '+label+' | '+value(name)+' | '+' | '.join(f"{result['slices'][part]['macro_f1'][name]:.4f}".replace('.',',') for part in ('empty_original','quotes','short_history'))+' |')
    lines += ['', '**Первичное сравнение:** внутренне выбранное усреднение '+value('selected_mean_bootstrap_fine')+' против контроля '+value('baseline_mean_bootstrap_fine')+'.','',
        '| Парное изменение | Условный 95% интервал macro-F1 |','|---|---|']
    for key,ci in result['comparisons_95'].items():
        if 'bootstrap_fine' in key:
            left,right = key.split('_minus_')
            descriptions = {**LABELS,'old_single':'Прежняя совместная модель, один seed'}
            lines.append(f"| {descriptions.get(left,left)} − {descriptions.get(right,right)} | [{ci['low']:+.5f}; {ci['high']:+.5f}] |")
    lines += ['', 'Интервалы получены 500 парными bootstrap-перевыборками внешних авторов для сохранённых прогнозов. Они не учитывают весь разброс обучения, многократное исследование этих development-разбиений и выбор гипотез по прежним результатам. Поправка на множественные сравнения не применялась.','',
        'Сохранённая дисперсия — мера сохранения геометрии эмбеддингов, а не доля полезного для метки смысла. Рост размерности оценивается по внешней метрике; он сам по себе не доказывает улучшение понимания текста. Цитата и документ здесь добавлены вместе, поэтому их индивидуальный вклад этим опытом не разделён.','',
        '### Воспроизводимость','',
        'Пройдены 45 тестов. Проверены четыре непересекающиеся роли авторов, происхождение PCA, выбор варианта по внутренним данным, равные веса seed, нормировка вероятностей и воспроизведение всех 36 внешних прогнозов из сохранённых вероятностей и множителей. Каждый новый классификатор повторно загружен и проверен при обучении. Контрольные суммы артефактов сохранены в `results/separated_research/artifact_audit.json`.','',
        'Результаты: `results/separated_research/comparison.json`, `separated_results.png`, модели и схемы в `fold_*/`. Запуск/продолжение: `train_separated_research.cmd`; статус: `check_status.cmd`. Прежние результаты сохранены.','']
    position = lines.index('### Воспроизводимость')
    extra = ['### Выводы этого этапа','']
    for name in ('fields64_s0_bootstrap_fine','e5_128_s0_bootstrap_fine','e5_256_s0_bootstrap_fine','selected_mean_bootstrap_fine'):
        reference = 'baseline_mean_bootstrap_fine' if name.startswith('selected') else 'base64_s0_bootstrap_fine'
        ci = result['comparisons_95'][name+'_minus_'+reference]
        verdict = ('условный интервал разницы целиком выше нуля' if ci['low']>0 else
                   'условный интервал разницы целиком ниже нуля' if ci['high']<0 else
                   'интервал включает ноль, преимущество не подтверждено')
        extra += [f"- {LABELS[name]}: {value(name)} против {value(reference)}; {verdict}."]
    extra += ['', 'Новый контроль трёх seed: '+value('baseline_mean_bootstrap_fine')+'; прежнее лучшее усреднение: '+value('old_best')+'. Сравнение относится ко всему изменению внутреннего протокола, а не только к независимости ранней остановки. Прежний результат сохранён; ни один вариант не объявляется окончательным победителем на официальном test.','',
        '### Диагностика полей и вероятностей','',
        '| Группа | Постов | Авторов | Контроль 64, один seed | + цитата/документ, один seed |',
        '|---|---:|---:|---:|---:|']
    for field,entry in diagnostic['slices'].items():
        extra.append(f"| {field} | {entry['posts']} | {entry['authors']} | {entry['macro_f1']['base64_s0_bootstrap_fine']:.4f} | {entry['macro_f1']['fields64_s0_bootstrap_fine']:.4f} |")
    extra += ['', 'Эти срезы исследовательские; малое число документов ограничивает устойчивость выводов.','',
        '| Один seed, до множителей классов | Log loss ↓ | Brier ↓ | Macro-F1 argmax |','|---|---:|---:|---:|']
    for name,entry in diagnostic['probability_quality'].items():
        extra.append('| '+name+' | '+' | '.join(f'{entry[key]:.4f}' for key in ('log_loss','brier','raw_argmax_macro_f1'))+' |')
    extra += ['','Условные авторские bootstrap-интервалы изменения на срезах:','']
    for name,entry in diagnostic['slices'].items():
        ci = entry['fields_minus_base_95']
        extra += [f"- {name}: [{ci['low']:+.5f}; {ci['high']:+.5f}]."]
    extra += ['', 'Перспективный локальный сигнал — посты с текстом документа: 0,6206 → 0,6480 на 781 посте 301 автора. Это исследовательский срез после просмотра результатов, без поправки на множественные сравнения. Цитата и документ добавлялись вместе; следующий отдельный эксперимент должен разделить их вклад, а не автоматически заменять общую модель этим вариантом.']
    extra += ['', 'Таблица отделяет качество вероятностей от настройки решения. Суммы важности дополнительных признаков CatBoost сохранены в `diagnostics.json`: это описательная характеристика использования признаков, не причинная оценка их пользы.','',
        'Методические источники: [настройка решающего порога](https://scikit-learn.org/stable/modules/classification_threshold.html), [PCA и объяснённая дисперсия](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html). Разделение четырёх ролей и вложенное расширение реализованы в этом проекте.','']
    lines[position:position] = extra
    path = ROOT/'IMPROVEMENT_PLAN_RU.md'
    content = path.read_text(encoding='utf-8')
    heading = lines[0]
    if heading in content:
        start = content.index(heading)
        end = content.find('\n## ',start+len(heading))
        content = content[:start]+'\n'.join(lines)+(content[end:] if end>=0 else '')
    else:
        position = content.index('\n## ')
        content = content[:position]+'\n'+'\n'.join(lines)+content[position:]
    content = content.replace('Лучшее усреднение нового этапа —','Лучшее усреднение предыдущего этапа —',1)
    path.write_text(content,encoding='utf-8')


def main():
    result = json.loads((OUT/'comparison.json').read_text())
    audit(result)
    plot(result)
    write_report(result,diagnostics())
    print(json.dumps(dict(audit='passed',selected=result['oof']['selected_mean_bootstrap_fine']['macro_f1'],baseline=result['oof']['baseline_mean_bootstrap_fine']['macro_f1'])))


if __name__=='__main__':
    main()
