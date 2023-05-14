# english_sub_prediction_app

Данный проект направлен на формирование модели, определяющей уровень английского на основании субтитров.
1. создан единый датасет с субтитрами
2. проведен EDA, включающий обработку текста в части восклицаний, приведения к нижнему регистру, стоп-слов и т.д., также была проведена лемматизация.
3. определен вид задачи - мультиклассификация, метрика F1_score
4. по итогу метрики наилучшая модель - логичестическая регрессия

По итогу был создан дамп модели и составлено небольшое приложение, которое при загрузке субтитров определяет уровень модели.

Для создания приложения был использован streamlit. Предобработка была проведена при помощь spacy, модели - sklearn и catboost.
