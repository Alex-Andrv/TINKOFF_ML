*P.S. все вычисления проводились под Mac m1 pro, поэтому нет гарантий что-то у вас сработает :)*


### **Придумайте образ и предпочтения инопланетян**

Эти инопланетяне - стартаперы с уникальными навыками в информационных технологиях, квантовой физике, биологии и математике, преданные исследованиям и созданию проектов, направленных на межгалактическую оптимизацию. Их характер можно охарактеризовать следующим образом:

1. Никакой прокрастинации: Они избегают бесполезного откладывания задач и всегда сосредотачиваются на своих целях.

2. Бесстрашие и уверенность: Им свойственна бескомпромиссная уверенность в своих способностях и отсутствие сомнений.

3. Фокус на работе: Они не отвлекаются на телевизор и минимизируют время для сна, поскольку они постоянно готовы переключаться между задачами, оставляя разным частям своего мозга возможность отдыхать.

4. Чистоплотность и комфорт: Инопланетяне-стартаперы ценят чистоту и комфорт, особенно в своем рабочем пространстве и ванной комнате.

5. Клятва межгалактическому совету: Они приняли клятву перед Межгалактическим Советом, обязывающую их посвятить свои знания и усилия в служении межгалактической оптимизации, независимо от других соблазнов и обстоятельств.

Клятва инопланетянина:

"Я обещаю посвятить свою жизнь и мои навыки оптимизации во имя блага всей космической общности. Я не буду брать жену, захватывать планеты, уничтожать цивилизации, иметь детей, или стремиться к короне и славе. Я останусь верен своему миссионному пути, буду стражем во тьме, защитником межгалактических проектов и щитом, оберегающим царство знаний и оптимизации. Я предаю свою жизнь и честь межгалактическому совету на ночь сейчас и на все грядущие времена"

<img src="2023-09-08 11.28.30.jpg" width="400" height="400" />

### Изучение датасета
#### Что такое image captioning? Почему над этим работают? Как формулируется задача?

Image captioning (подпись изображения) - это задача в области компьютерного зрения и обработки естественного языка, в которой компьютерная система создает текстовое описание для изображения. Основная цель image captioning - научить компьютер понимать содержание изображения и генерировать краткое и информативное описание, которое человек может прочитать.

Задачи, которые можно решать с помощью image captioning:

1. Автоматическая подпись изображений: Генерация текстовых описаний для фотографий или изображений, что может быть полезно для создания подписей в фотоальбомах или визуальных поисковых системах.

2. Помощь лицам с ограниченными возможностями: Image captioning может использоваться для описания изображений веб-сайтов, социальных сетей и т.д., чтобы обеспечить доступность контента для людей с нарушениями зрения.

3. Робототехника и автономные устройства: Роботы и автономные устройства могут использовать image captioning для понимания окружающей среды и принятия более информированных решений.

4. Поиск по изображению: Позволяет пользователю искать изображения на основе текстовых запросов.

5. Анализ медицинских изображений: Image captioning может помочь врачам в описании и интерпретации медицинских изображений, таких как рентгеновские снимки или снимки МРТ.

Важность image captioning заключается в улучшении взаимодействия между компьютерами и людьми в области визуального контента. Эта технология делает изображения более доступными, обеспечивает автоматизацию процесса создания подписей и увеличивает возможности роботов и автономных устройств для восприятия окружающей среды. Кроме того, image captioning имеет потенциал в медицинских и научных приложениях, что может улучшить точность и эффективность работы профессионалов в этих областях.
Отдельно стоит обратить внимание на 3 пункт. Весть Image captioning построен на стыке NPL и CV. Основной Chalange в том, чтобы подружить CV модели и NLP (например с помощью системы обучения CLIP). Сама идея - научить понимать NLP модели эмбединги изображений, может в ближайшем будущем помочь нам создать автономных роботов, не на базе RL или дифференциальный вычислений, а на базе CV и NLP. Например: [palm-saycan](https://sites.research.google/palm-saycan), уже творят магию.

Постановка задачи:

При заданном входном изображении I целью является создание описания C, описывающего визуальное содержимое, присутствующее на данном изображении, где C представляет собой набор предложений C = {c1, c2, ..., cn}, где каждое ci - это предложение в созданной подписи C. [[Taraneh et al., 2023](https://arxiv.org/pdf/2201.12944.pdf)]


#### Обзор датасета. Что представлено на изображениях?
На изображениях в Hotels-50K датасете представлены изображения номеров отелей и их интерьеров. Датасет был собран для идентификации отеля на фотографиях. Данная задача имеет важное значение для исследований по борьбе с торговлей людьми [[Abby et. all., 2019](https://www.researchgate.net/publication/330775269_Hotels-50K_A_Global_Hotel_Recognition_Dataset)]. В исходной статье к исходным изображаниям добавляют образы людей, нам же это не нужно. Поэтому можно взять только исходные изображения. 
Помимо этого, в архиве есть различные таблички [dataset.tar.gz](https://github.com/GWUvision/Hotels-50K/blob/master/input/dataset.tar.gz). *PS ребята залили платформо специфичные mac OS файлы в архив, потому нормально разархивировать датасет на mac больно, никогда не заливайте платформа специфичные файлы в архив*
Метаданные для отелей:
1. chain_info.csv: chain_id, chain_name
2. hotel_info.csv: hotel_id, hotel_name, chain_id, latitude, longitude
3. train_set.csv: image_id, hotel_id, image_url, image_source, upload_timestamp
4. test_set.csv: image_id, hotel_id, image_url, image_source, upload_timestamp
#### Сколько объектов в датасете? Сколько уникальных классов? Сбалансирован ли датасет?
1. Количество уникальных франшиз(классов): 93
2. Количество уникальный отелей: 50000
3. Распределение количества отелей во франшизе не равномерное. Очень похоже на экспоненциальное или логнормальное. Данные гипотезы не подтвердились статистическими тестами. 
4. Количество обучающех примеров: 1124215. Данные распределены не равномерно
5. Количество тестовых примеров: 16172. Данные распределены не равномерно

#### Какие параметры у изображений? Размер фотографий?
Весь датасет весит примерно 100 гб. Во-первых, я это не скачаю за разумное время, во-вторых, не хватит диска, в-третьих, это не нужно для нашей задачи. Поэтому скачаем фотографии 10 отелей. Для этого воспользуемся [download_train.py](eda_lib%2Fdownload_train.py) скриптом, основа которого была взята из репозитория с датасетом. Однако, его пришлось менять.
Данный скрипт меняет размер фото. Суть изменений заключается в том, что мы приводим размеры скачанных фотографий к стандартным значениям "640:Y" или "X:640", где X и Y - это значения меньше 640 пикселей, в зависимости от того, является ли фотография горизонтальной или вертикальной.

Количество уникальных размеров 46. Всего фото 406. 

### Обогащение датасета описаниями

#### Составьте и предобработайте тренировочную выборку датасета. Опишите, какую предобработку вы использовали и почему.
Частично на данный вопрос я дал ответ в предыдущем пункте. Я взял 10 рандомных отелей и скачал все доступные фотографии, связанные с ними. В исходной статье про сlip, описан способ предобработки изображений.
Для передачи изображений в кодировщик Transformer каждое изображение разбивается на последовательность фиксированного размера неперекрывающихся участков, которые затем линейно встраиваются. Введен символ [CLS], который служит представлением всего изображения. Авторы также добавляют абсолютные позиционные вложения и передают полученную последовательность векторов стандартному кодировщику Transformer. Класс CLIPImageProcessor с дефолтными настройками как раз этим и занимается, поэтому нет особого смысла писать свой препроцессор.
Кроме того, я использовал специальный шаблон:  "a photo of a {label}". Многие исследователи заметили значительное улучшение оценки точности при использовании этого формата. [[A. Radford, et. al.](https://arxiv.org/abs/2103.00020)]

#### Обработайте каждую фотографию из выборки, добавьте к ней описание. Проанализируйте полученные результаты
См. [hotels_50s_eda.ipynb](hotels_50s_eda.ipynb) ячейка 35. 
Алгоритм тегирования:
1. Придумываем 20 тегов
2. Запизиваем их в шаблон:  "a photo of a {label}". И переводим в эмбединги
3. Для каждой фотографии вы создаете ее эмбеддинг. Затем вы вычисляете косинусное расстояние между эмбеддингами текста и эмбеддингами каждой фотографии. Вы хотите вернуть теги, для которых косинусное расстояние больше 0,25. Если таких нет, то вы хотите вернуть тег с наибольшим косинусным расстоянием. 
Результаты:
1. Cреднее количество тегов для одной фотки: 2.1575091575091574
2. Cреднее косинус у топ 1 фотки: 0.2661934792995453
3. Cреднее разница косинусов между 1 и 10 тегом: 0.05291054770350456
4. Cреднее количество уникальных тегов у отеля: 10.461538461538462
5. Максимальное количество тегов у отеля: 18
6. Минимальное количество тегов у отеля: 2
Cкажем прямо, косинус далек от 1. Более того, разница между 1 тегом и 10, не такая уж и большая :0 А вот как такое улучшить без обучения - очень трудный вопрос. Я не нашел ничего приличного в статьях.

#### Попробуйте кластеризовать датасет по тэгам. Сходится ли число классов и число кластеров? Почему? 
Используем алгоритм k-modes. См. [hotels_50s_eda.ipynb](hotels_50s_eda.ipynb). 
Действительно, многие алгоритмы кластеризации представляют собой эвристические методы, которые не всегда точно отражают реальную структуру данных. Выбор оптимального числа кластеров (параметр k) может быть нетривиальным. Однако, на мой взгляд, более существенной проблемой является неполный список тегов. При разработке этого списка я стремился включить наиболее общие характеристики, применимые к отелям, такие как "кровать", "стул", "стол", "шторы". Однако ограничиваясь только этими тегами, становится сложно провести качественную кластеризацию.

### Изменение изображения при помощи диффузионной модели
Устанавливаем приложение по [инструкции](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Installation-on-Apple-Silicon)
#### Сделайте выборку из 5-10 фотографий из датасета. Постарайтесь отобрать как можно более различные и интересные примеры **(1балл)**;
См. [hotels_50s_eda.ipynb](hotels_50s_eda.ipynb) 
#### Сделайте серию правок изображений, стараясь каждый раз менять лишь один конкретный параметр на всей выборке из предыдущего этапа. Например, только время суток или только работу электроприборов **(2 балла)**;
См. [hotels_50s_eda.ipynb](hotels_50s_eda.ipynb) 
####  Подумайте как можно улучшить изменение не затрагивая остальные элементы изображения. Приведение в отчёте ссылки на решения и статьи по этому направлению приветствуются **(2 балла)**;
Вообще все это очень долгий процесс, а время у меня закончилось....
1. В общем очень трудно подобрать гиперпараметры для двух фоток, не говоря уже о 5.
2. Можно придумать универсальный промт, но такие параметры как "steps", "denoising_strength" нужно подбирать для каждой картинки(и не забыть зафиксировать сид).
3. denoising_strength - влияет на силу изменения изображения, поэтому не стоит сильно увеличивать данный параметр.
4. Самые крутые промты - просто слова. Модель работает хуже, если писать подробные промты
5. Забавно, что снег с первой итерации удалился везьде.
#### Напишите небольшой отчет по этой части в Readme репозитория о проведенных экспериментах. Что получилось? Что нет? Введёт лиинопланетян такая картинка? **(1 балл)**;
Что не получилось:
1. Маленький косинус у топ 1 тега, даже если он очень хорошо описывает картинку.
2. Косинусы между тегами, которые присутствуют на картинке, и теми, которых на ней нет, очень близки.
3. Плохая кластеризация. Основная проблема в том, что придумал слабые теги 
4. Ну и очень трудно подобрать стабильные гиперпараметры для stable-diffusion, чтобы результаты были +- предсказуемыми.
Что получилось:
1. Все-таки я доделал 
2. Получилось разобраться в принципах работы CLIP и stable-diffusion. Изучил некоторые статьи по улучшению качество zero shot image classification, некоторые очень подробно
3. Примерно понимаю, как гипотетически можно повысить качество классификации
4. В конце получились не плохие фотки так-то.
5. Узнал про ControlNet, вполне вероятно его использование сильно бы упростило мне задачу