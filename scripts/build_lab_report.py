"""Build the Russian research report from verified saved metrics, without training."""
from pathlib import Path
import json
import hashlib
from xml.sax.saxutils import escape
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT,TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak,Table,TableStyle,KeepTogether
from reportlab.graphics.shapes import Drawing,String,Line,Circle,Rect,PolyLine

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output'/'pdf'
OUT.mkdir(parents=True,exist_ok=True)
pdfmetrics.registerFont(TTFont('Arial',r'C:\Windows\Fonts\arial.ttf'))
pdfmetrics.registerFont(TTFont('ArialBold',r'C:\Windows\Fonts\arialbd.ttf'))
pdfmetrics.registerFontFamily('Arial',normal='Arial',bold='ArialBold',italic='Arial',boldItalic='ArialBold')
INK=colors.HexColor('#17252C'); MUTED=colors.HexColor('#526570'); BLUE=colors.HexColor('#326E93')
TEAL=colors.HexColor('#268579'); GOLD=colors.HexColor('#AE772A'); GRID=colors.HexColor('#D9E2E7')
W=479
styles={
 'title':ParagraphStyle('Title',fontName='ArialBold',fontSize=27,leading=32,textColor=colors.black,spaceAfter=17),
 'h1':ParagraphStyle('Heading1',fontName='ArialBold',fontSize=20,leading=25,textColor=colors.black,spaceAfter=15),
 'h2':ParagraphStyle('Heading2',fontName='ArialBold',fontSize=12.5,leading=17,textColor=colors.black,spaceBefore=10,spaceAfter=6),
 'body':ParagraphStyle('Body',fontName='Arial',fontSize=10.3,leading=15.3,textColor=INK,spaceAfter=9),
 'lead':ParagraphStyle('Lead',fontName='Arial',fontSize=12,leading=18,textColor=INK,spaceAfter=14),
 'small':ParagraphStyle('Small',fontName='Arial',fontSize=8.6,leading=12.3,textColor=MUTED,spaceAfter=8),
 'cell':ParagraphStyle('Cell',fontName='Arial',fontSize=9,leading=12,textColor=INK),
 'headcell':ParagraphStyle('HeaderCell',fontName='ArialBold',fontSize=9,leading=12,textColor=colors.white),
}
sources={}
def read(name):
    p=ROOT/'results'/f'{name}.json'
    sources[str(p.relative_to(ROOT)).replace('\\','/')]=hashlib.sha256(p.read_bytes()).hexdigest()
    return json.loads(p.read_text(encoding='utf-8'))
lecture=read('lecture_models/decision_audit/comparison')
semantic=read('semantic/comparison'); visual=read('visual/comparison'); history=read('semantic_history/comparison')
priority=read('priority_research/comparison'); stability=read('stability_research/comparison')
new=read('separated_research/comparison'); diag=read('separated_research/diagnostics')
baseline=read('baseline'); test=read('maria/test_metrics')
final_test=read('official_test_v3/test_metrics')
final_protocol=read('official_test_v3/frozen_protocol')
audit=read('separated_research/artifact_audit')
read('empty_text/comparison')
for relative in ('README.md','IMPROVEMENT_PLAN_RU.md'):
    sources[relative]=hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()
assert new['complete'] and audit['passed'] and new['rows']==71395 and new['classifiers_trained']==44
def score(data,name): return data['oof'][name]['macro_f1']
def num(v,n=4): return f'{v:.{n}f}'.replace('.',',')
def P(text,kind='body'): return Paragraph(text,styles[kind])
story=[]
def p(text,kind='body'): story.append(P(text,kind))
def heading(text): p(text,'h2')
def page(title):
    if story: story.append(PageBreak())
    p(title,'h1')
def caption(text): p(text,'small')
def table(headers,rows,widths):
    cells=[[P(escape(str(c)),'headcell') for c in headers]]+[[P(escape(str(c)),'cell') for c in row] for row in rows]
    t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),BLUE),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F2F6F8')]),
        ('GRID',(0,0),(-1,-1),.45,colors.HexColor('#D9D9D9')),('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
    story.extend([t,Spacer(1,11)])
def text(d,x,y,s,size=9,color=INK,font='Arial',anchor='start'):
    d.add(String(x,y,s,fontName=font,fontSize=size,fillColor=color,textAnchor=anchor))
def dotchart(labels,values,lo,hi,ticks,highlight=None,row=30):
    h=50+row*len(labels); d=Drawing(W,h); left=212; right=433
    def xx(v): return left+(v-lo)/(hi-lo)*(right-left)
    for tick in ticks:
        x=xx(tick); d.add(Line(x,31,x,h-13,strokeColor=GRID,strokeWidth=.6)); text(d,x,17,num(tick,3),8,MUTED,anchor='middle')
    for i,(label,value) in enumerate(zip(labels,values)):
        y=h-25-row*i
        text(d,0,y-3,label,9.5)
        color=TEAL if i==highlight else BLUE
        d.add(Circle(xx(value),y,3.8,fillColor=color,strokeColor=color))
        text(d,xx(value)+7,y-3,num(value),9,font='ArialBold')
    text(d,(left+right)/2,1,'OOF macro-F1',8,MUTED,anchor='middle')
    return d
def role_diagram():
    d=Drawing(W,95)
    roles=[('Обучение','Модели и PCA','1 756 авторов'),('Остановка','Число деревьев','220 авторов'),('Выбор решения','Правило и вариант','220 авторов'),('Внешняя оценка','Только метрики','1 099 авторов')]
    for i,items in enumerate(roles):
        x=i*122
        d.add(Rect(x,18,113,65,fillColor=colors.white,strokeColor=GRID,strokeWidth=.8))
        for j,label in enumerate(items): text(d,x+56.5,65-j*17,label,8.6,INK if j==0 else MUTED,'ArialBold' if j==0 else 'Arial',anchor='middle')
    return d
def variance_chart():
    d=Drawing(W,158); left=50; right=425; bottom=28; top=139
    for v in (50,60,70,80):
        y=bottom+(v-50)/30*(top-bottom);d.add(Line(left,y,right,y,strokeColor=GRID,strokeWidth=.6));text(d,left-10,y-3,str(v)+'%',8,MUTED,anchor='end')
    for fold,color in zip(('1','2','3'),(BLUE,TEAL,GOLD)):
        pts=[]
        for i,n in enumerate((64,128,256)):
            x=left+i*(right-left)/2;v=100*new['pca'][fold]['explained_variance'][str(n)];y=bottom+(v-50)/30*(top-bottom)
            pts.extend((x,y));d.add(Circle(x,y,2.8,fillColor=color,strokeColor=color))
        d.add(PolyLine(pts,strokeColor=color,strokeWidth=1.4))
    for i,n in enumerate((64,128,256)): text(d,left+i*(right-left)/2,10,str(n),9,MUTED,anchor='middle')
    text(d,235,0,'Число компонент E5',8,MUTED,anchor='middle')
    return d
def forest():
    entries=[]
    for label,key in [('Отдельные поля','fields64_s0_bootstrap_fine_minus_base64_s0_bootstrap_fine'),('128 компонентов','e5_128_s0_bootstrap_fine_minus_base64_s0_bootstrap_fine'),('256 компонентов','e5_256_s0_bootstrap_fine_minus_base64_s0_bootstrap_fine')]:
        ci=new['comparisons_95'][key];name=key.split('_minus_')[0]
        entries.append((label,score(new,name)-score(new,'base64_s0_bootstrap_fine'),ci))
    d=Drawing(W,178); left=175;right=440;lo=-.005;hi=.003
    def xx(v):return left+(v-lo)/(hi-lo)*(right-left)
    for v in (-.004,-.002,0,.002):
        x=xx(v);d.add(Line(x,33,x,163,strokeColor=MUTED if v==0 else GRID,strokeWidth=.8));text(d,x,20,f'{v:+.3f}'.replace('.',','),8,MUTED,anchor='middle')
    for i,(label,mean,ci) in enumerate(entries):
        y=144-i*43;text(d,0,y-3,label,10);d.add(Line(xx(ci['low']),y,xx(ci['high']),y,strokeColor=BLUE,strokeWidth=2));d.add(Circle(xx(mean),y,3.7,fillColor=TEAL,strokeColor=TEAL))
    text(d,305,3,'Изменение macro-F1 относительно контроля',8,MUTED,anchor='middle')
    return d

# 1  Summary and a development-only trajectory
p('Прогноз относительного успеха публикаций','title')
p('Лабораторная работа 1<br/>Social Media Analytics','lead')
p('Maria Sushcheva • Kairos Dataset • 27 октября 2026','small')
p('Исследование текста, изображений и истории автора','h2')
p('Лучший наблюдавшийся результат разработки составил <b>0,6665 macro-F1</b>. Его дала порядковая модель с историей цитирования, усреднением трёх запусков и настройкой правила выбора класса. Последний этап с раздельными внутренними группами не превзошёл этот результат.','lead')
labels=['Исходное объединение','CatBoost и SVM','Семантика текста E5','Изображения SigLIP 2','Смысловая история автора','Порядковая модель','Усреднение и правило решения']
vals=[score(lecture,'v1_fusion'),score(lecture,'catboost_svm_fusion'),score(semantic,'semantic_fusion'),score(visual,'multimodal_catboost'),score(history,'visual_neighbors'),score(priority,'ordinal'),score(stability,'joint_mean_bootstrap_fine')]
story.append(dotchart(labels,vals,.625,.673,[.63,.64,.65,.66,.67],6,28))
caption('Рисунок 1. Этапы разработки на одних внешних разбиениях официального train. Это последовательные исследовательские результаты, а не независимые доказательства вклада каждого изменения. Источники [2–7].')
p('На официальном test замороженный ансамбль получил <b>'+num(final_test['macro_f1'])+'</b> против <b>0,6229</b> у первоначальной модели и <b>0,5410</b> у baseline. OOF и test показаны отдельно: это разные оценки.','body')

# 2  Task, sample and roles
page('Данные и схема проверки')
p('Задача состоит в определении того, оказался ли отклик на публикацию ниже, около или выше обычного уровня её автора. Одинаковое число реакций может означать разный успех у разных авторов.')
p('<b>Уровень автора</b> = медиана log(1 + реакции) предыдущих собственных постов.<br/><b>Отклонение поста</b> = log(1 + реакции поста) − уровень автора.')
table(['Класс','Условие для отклонения','Постов в train'],[['below','Меньше −0,549', '14 225'],['typical','Между порогами','42 892'],['above','Больше 1,157','14 278']],[90,260,129])
p('В официальном train находятся <b>71 395 постов 3 295 авторов</b>, в test — 17 623 поста 824 авторов. Первые пять собственных публикаций формируют историю и не имеют метки. Основная метрика macro-F1 одинаково учитывает три класса; QWK дополнительно учитывает порядок классов. [1]')
heading('Внешняя оценка по новым авторам')
p('Внутри train использованы три фиксированных разбиения по авторам. Для каждого поста OOF-прогноз получен моделью, не обучавшейся на его авторе. Общий macro-F1 рассчитан по объединённым прогнозам, а не как среднее трёх F1.')
story.append(role_diagram())
caption('Рисунок 2. Четыре непересекающиеся роли авторов в последнем эксперименте. Показаны размеры первого разбиения. В двух других: 1 757 / 220 / 220 / 1 098. [8]')
p('В прежних этапах одна внутренняя группа использовалась и для ранней остановки, и для настройки решения. В последнем этапе она разделена пополам. Поэтому сравнение протоколов отражает также уменьшение объёма данных для каждой настройки.')
p('<b>Защита от утечек.</b> ID авторов и постов, текущие реакции и целевое отклонение не передаются модели. PCA обучается только на fitting-авторах. История содержит строго более ранние публикации; совпадения времени исключены. Исторические счётчики при этом являются снимком на дату сбора, а не точной записью знаний в момент публикации.','small')

# 3  Learning and lecture methods
page('Сравнение методов обучения')
p('Использованы линейный SVM, наивный Байес, случайный лес и градиентный бустинг CatBoost. Проверены объединение текстовых и табличных признаков и усреднение прогнозов. Классификаторы обучались с учителем на метках below, typical и above. [2]')
names=['linear_svm','naive_bayes','random_forest','catboost_400','catboost_800','catboost_svm_fusion']
story.append(dotchart(['TF-IDF и линейный SVM','TF-IDF и наивный Байес','Случайный лес','CatBoost 400 деревьев','CatBoost до 800 деревьев','CatBoost и SVM'],[score(lecture,n) for n in names],.4,.68,[.4,.45,.5,.55,.6,.65],5,29))
caption('Рисунок 3. Сравнение обученных классификаторов. SVM и Байес используют текст, лес — таблицу признаков, CatBoost — оба источника. Сравнение не изолирует качество алгоритма от качества входов. [2]')
heading('Как учитывался смысл текста')
p('Замороженный multilingual-e5-base преобразует собственный и расширенный текст в векторы. Длинные тексты разбиваются на фрагменты; векторы агрегируются. Слово «замороженный» относится к E5: классификаторы поверх его представлений обучались заново.')
table(['Представление и классификатор','OOF macro-F1'],[['E5 и логистическая регрессия',num(score(semantic,'e5_logistic'))],['E5 и нейросеть MLP',num(score(semantic,'e5_mlp'))],['E5 и таблица признаков с MLP',num(score(semantic,'e5_history_mlp'))],['E5 и таблица признаков с CatBoost',num(score(semantic,'e5_history_catboost'))]],[350,129])
p('Семантика оказалась полезнее в сочетании с историей и структурой публикации. Одна только замена словарного текста на эмбеддинги не решила задачу. В основных моделях объединённые E5-векторы сокращались PCA до 64 компонент. [3]')

# 4  Multimodal context
page('Изображения и смысловая история')
p('Для визуального представления использован замороженный SigLIP 2. Из контента исключены аватары и логотипы, повторяющиеся ссылки на один файл удалены. Для нескольких изображений сохранялись агрегированный вектор и представление первого изображения.')
table(['Источник','Объём в официальном train'],[['Посты с контентными изображениями','33 205  ·  46,5%'],['Посты с видео','4 798  ·  6,7%'],['Посты без собственного текста','5 726  ·  8,0%']],[290,189])
p('Визуальный эксперимент использует изображения, обложки и превью из core. Полное моделирование видеоряда не выполнялось. Загрузка всех видео не означает, что модель уже обучалась на каждом кадре.')
story.append(dotchart(['E5 и история','Только добавление метаданных','Добавление визуальных векторов','Смысловые соседи текста','Соседи текста плюс изображения'],[score(visual,'e5_history_catboost'),score(visual,'media_metadata_catboost'),score(visual,'multimodal_catboost'),score(history,'text_neighbors'),score(history,'visual_neighbors')],.648,.666,[.65,.655,.66,.665],4,29))
caption('Рисунок 4. Развитие мультимодальной модели. Последние два варианта относятся к следующему исследованию истории; этапы не являются единым факторным экспериментом. [4, 5]')
heading('Что означает смысловая история')
p('Для нового поста рассчитаны сходство с прежними публикациями автора, отклонение от его среднего смыслового вектора и признаки ближайших прошлых постов. Это позволяет различать привычную тему и необычную для автора публикацию.')
p('Кандидат v2 достиг <b>0,6605 macro-F1</b>. На следующем этапе история того же режима цитирования дала 0,6628, а порядковая модель — 0,6630. Экспоненциальное взвешивание истории и адаптивное объединение веток не стали лучшими вариантами. [5, 6]')

# 5  Stability and decision mechanism
page('Порядковая модель и выбор класса')
p('Порядковая модель состоит из двух бинарных CatBoost: первый оценивает вероятность класса выше below, второй — выше typical. Если оценки нарушают порядок, они согласуются. Из двух вероятностей строится распределение по трём классам. Это собственная реализация двух порогов, не обучение CORAL.')
heading('Настройка решения поверх одинаковых вероятностей')
p('Вероятности трёх seed усреднялись с равными весами. Затем выбирались множители классов below и above при множителе typical = 1. Плотная сетка содержит 529 пар. В bootstrap-варианте правило выбиралось на 100 перевыборках внутренних авторов, а множители усреднялись геометрически.')
rules=['grid','fine_grid','bootstrap_grid','bootstrap_fine']
story.append(dotchart(['Прежняя сетка','Плотная сетка','Bootstrap прежней сетки','Bootstrap плотной сетки'],[score(stability,'joint_mean_'+r) for r in rules],.662,.669,[.662,.664,.666,.668],3,32))
caption('Рисунок 5. Порядковая модель с историей цитирования, три seed. Вероятности во всех четырёх строках одинаковы: меняется только правило выбора класса. [7]')
p('Лучший наблюдавшийся вариант дал <b>0,6665 macro-F1</b> и <b>0,6030 QWK</b>. Изменение относительно прежней сетки составляет около +0,0023 macro-F1; условный 95% интервал разницы — от +0,00093 до +0,00381.')
heading('Почему усреднение не гарантирует выигрыш')
p('Seed меняет случайность обучения, но не создаёт новое независимое разбиение данных. Для порядковой модели с цитированием отдельные seed дали 0,6625, 0,6627 и 0,6636. Выбор самого удачного seed по внешним авторам не использовался.')
p('Лучшее значение 0,6665 было найдено в процессе разработки на одних и тех же внешних разбиениях. Последующая оценка замороженного ансамбля на официальном test приведена отдельно. Bootstrap-интервалы не учитывают весь многократный выбор гипотез. [7, 9]','small')

# 6  Latest experiment
page('Раздельные поля и размерность E5')
p('В последнем этапе разделены авторы для ранней остановки и настройки решения. Обучены <b>44 новых бинарных классификатора</b>. Четыре варианта сначала проверены с одним фиксированным seed; затем контроль и выбранный внутри каждого разбиения вариант усреднены по трём seed. [8]')
latest=['base64_s0_bootstrap_fine','fields64_s0_bootstrap_fine','e5_128_s0_bootstrap_fine','e5_256_s0_bootstrap_fine']
story.append(dotchart(['Контроль E5 64','64 плюс цитата и документ','E5 128','E5 256'],[score(new,n) for n in latest],.66,.668,[.66,.662,.664,.666,.668],1,28))
caption('Рисунок 6. Сопоставление одного seed на одинаковом новом протоколе. Цитата и документ кодируются отдельно и сокращаются до 16 компонент каждый. [8]')
p('Расширения до 128 и 256 сохраняют исходные 64 компоненты без изменений. Новые направления получены PCA остатка реконструкции и обучены только на fitting-авторах. Это вложенные расширения, а не независимое переобучение PCA с нуля.')
story.append(variance_chart())
caption('Рисунок 7. Сохранённая дисперсия E5 выросла примерно с 54–55% до 65–66% и 79%. Три линии соответствуют трём разбиениям. Дисперсия описывает геометрию векторов, а не долю полезного для метки смысла. [8]')
p('При трёх seed контроль получил <b>0,6644</b>, а процедура внутреннего выбора — <b>0,6639</b>. На разбиениях выбраны соответственно 128 компонент, отдельные поля и контроль 64. Прежний лучший результат 0,6665 не превзойдён. Изменение внутреннего протокола одновременно уменьшило группы для настройки, поэтому снижение нельзя объяснять одной причиной.')

# 7  Uncertainty and slices
page('Надёжность улучшений и группы постов')
p('Для последних вариантов рассчитаны 500 парных bootstrap-перевыборок внешних авторов. Все посты автора перевыбираются вместе. Интервалы условны на уже обученных прогнозах и не включают весь разброс обучения или поправку на множественные сравнения.')
story.append(forest())
caption('Рисунок 8. Точка — разница округлённых OOF macro-F1, линия — 95% интервал на неокруглённых прогнозах. Если интервал включает ноль, преимущество не подтверждено. У 256 компонент верхняя граница лишь немного ниже нуля. [8]')
table(['Группа','Постов','Контроль 64','Отдельные поля'],[['Без собственного текста','5 726','0,4814','0,4776'],['С текстом цитаты','28 857','0,5961','0,5950'],['С текстом документа','781','0,6206','0,6480']],[220,65,95,99])
caption('Таблица. Один seed, одинаковый новый протокол и bootstrap-правило. Группы могут пересекаться. Текст цитаты и наличие пересылки — разные признаки; здесь используется именно наличие текста. [8]')
heading('Документы дают локальный положительный сигнал')
p('На 781 посте 301 автора прирост от отдельных полей составил около <b>+0,0274 macro-F1</b>; условный интервал — [+0,00596; +0,05105]. Это исследовательский срез после просмотра результатов. Цитата и документ добавлялись вместе, поэтому индивидуальный вклад документной ветки ещё не установлен.')
heading('Проблема пустого собственного текста остаётся')
p('Отдельные поля не улучшили эту группу. В прежнем эксперименте OCR и специальные правила для таких постов могли улучшать её метрику ценой общей. [10] Сначала нужно проверить отдельные ветки цитаты и документа и их маршрутизацию на внутренних авторах, сохраняя общую метрику как главный критерий.')

# 8  Official test evaluation
page('Результат на официальном test')
p('После завершения разработки выбран вариант joint_mean_bootstrap_fine. До расчёта тестовых метрик зафиксированы контрольные суммы моделей, преобразования PCA и множители классов. Параметры по test не подбирались. [9]')
table(['Модель','Macro-F1','QWK','Accuracy'],[
 ['Официальный baseline',num(baseline['results']['baseline']['macro_f1']),num(baseline['results']['baseline']['qwk']),num(baseline['results']['baseline']['accuracy'])],
 ['Первоначальная модель',num(test['macro_f1']),num(test['qwk']),num(test['accuracy'])],
 ['Замороженный ансамбль',num(final_test['macro_f1']),num(final_test['qwk']),num(final_test['accuracy'])]],[215,88,88,88])
p('Оценка выполнена на <b>17 623 постах 824 авторов</b>. Изменение macro-F1 относительно первоначальной модели: <b>'+f"{final_test['macro_f1']-test['macro_f1']:+.4f}".replace('.',',')+'</b>. Условный 95% авторский bootstrap-интервал macro-F1 ансамбля: ['+num(final_test['author_bootstrap_95']['low'])+'; '+num(final_test['author_bootstrap_95']['high'])+'].')
heading('Как получен единый тестовый прогноз')
p('Использованы 18 сохранённых бинарных CatBoost: три разбиения, три seed и два порядковых порога. В каждом разбиении усредняются вероятности трёх seed. Затем применяется его сохранённое правило классов, оценки нормируются и усредняются между тремя разбиениями с равными весами. Побеждает класс с максимальной итоговой оценкой.')
p('Переобучение на всём train не выполнялось. В OOF каждый автор оценивался одной группой моделей своего внешнего разбиения; на test объединены три группы. Поэтому тестовый ансамбль и процедура получения OOF различаются способом объединения, а 0,6665 не является гарантией тестового качества.')
heading('Качество по классам')
table(['Класс','Precision','Recall','F1','Постов'],[[name,num(final_test['by_class'][name]['precision']),num(final_test['by_class'][name]['recall']),num(final_test['by_class'][name]['f1-score']),str(int(final_test['by_class'][name]['support']))] for name in ('below','typical','above')],[111,92,92,92,92])
p('Прогнозы сохранены до чтения тестовых меток для оценки. Предыдущая модель уже оценивалась на этом test, поэтому это повторно использованный официальный тестовый набор проекта, а не новый скрытый набор. После получения результата новые настройки по нему не выбирались.','small')

# 9  Conclusions and reproducibility
page('Выводы и воспроизводимость')
p('<b>Основной вывод.</b> Для относительного успеха публикации важен контекст автора. Наиболее полезной оказалась совместная модель истории, семантики текста и изображений. Простое увеличение числа PCA-компонент не повысило качество в проверенных настройках.')
p('<b>Рабочий результат.</b> Лучшее прежнее усреднение имеет 0,6665 OOF macro-F1. Последний протокол с раздельной остановкой и выбором решения дал 0,6644 у контроля. Замороженный ансамбль сохранённых моделей получил <b>'+num(final_test['macro_f1'])+' macro-F1 на официальном test</b>.')
p('<b>Следующий обоснованный опыт.</b> Раздельно добавить только документ и только цитату. Проверить, сохраняется ли выигрыш на документах при нескольких seed и не ухудшается ли общий macro-F1. Выбор маршрутизации выполнять на внутренних авторах, не на внешних результатах.')
heading('Как воспроизвести последний этап')
p('Последний исследовательский этап запускается через <b>train_separated_research.cmd</b>, тестовая оценка — через <b>evaluate_test.cmd</b>. Состояние показывает <b>check_status.cmd</b>. Тестовые прогнозы, протокол и метрики находятся в <b>results/official_test_v3</b>. Повторный запуск использует сохранённый результат без нового подбора.')
p('Прошли <b>47 тестов</b>. Проверены роли авторов, происхождение PCA, равные веса seed, объединение тестовых прогнозов и восстановление исследовательских прогнозов. Генерация новых признаков воспроизвела обучающие признаки на контрольных авторах. Все данные обрабатывались локально; идентификация людей по кодам не выполнялась.')
heading('Источники данных и результатов')
refs=[
 '[1] README.md репозитория <link href="https://github.com/Danil482/kairos-dataset" color="#326E93">Kairos Dataset</link>; results/baseline.json; results/maria/test_metrics.json.',
 '[2] results/lecture_models/decision_audit/comparison.json.',
 '[3] results/semantic/comparison.json. Модель: <link href="https://huggingface.co/intfloat/multilingual-e5-base" color="#326E93">multilingual-e5-base</link>.',
 '[4] results/visual/comparison.json. Модель: <link href="https://huggingface.co/google/siglip2-base-patch16-224" color="#326E93">SigLIP 2 base patch16 224</link>.',
 '[5] results/semantic_history/comparison.json.',
 '[6] results/priority_research/comparison.json.',
 '[7] results/stability_research/comparison.json и diagnostics.json.',
 '[8] results/separated_research/comparison.json, diagnostics.json, artifact_audit.json и protocol.json.',
 '[9] results/official_test_v3/test_metrics.json, frozen_protocol.json и prediction_manifest.json.',
 '[10] results/empty_text/comparison.json; подробности исследований и объёмов контента в IMPROVEMENT_PLAN_RU.md.',
 'Методическая документация: <link href="https://catboost.ai/docs/en/references/training-parameters/common" color="#326E93">CatBoost</link>, <link href="https://scikit-learn.org/stable/modules/classification_threshold.html" color="#326E93">настройка решения</link>, <link href="https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html" color="#326E93">PCA</link>.',
]
for ref in refs:p(ref,'small')

def footer(canvas,doc):
    canvas.saveState()
    canvas.setFont('Arial',8);canvas.setFillColor(MUTED)
    canvas.drawString(58,30,'SOCIAL MEDIA ANALYTICS  /  KAIROS')
    canvas.drawRightString(A4[0]-58,30,str(doc.page))
    canvas.restoreState()

file=OUT/'Social_Media_Analytics_Sushcheva_Maria.pdf'
doc=SimpleDocTemplate(str(file),pagesize=A4,rightMargin=58,leftMargin=58,topMargin=45,bottomMargin=48,
    title='Прогноз относительного успеха публикаций',author='Maria Sushcheva',subject='Лабораторная работа 1 по Social Media Analytics')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
(OUT/'report_sources.json').write_text(json.dumps(dict(sources=sources,report_sha256=hashlib.sha256(file.read_bytes()).hexdigest()),indent=2),encoding='utf-8')
print(file)
