# People Detection on Video

Это система компьютерного зрения для трекинга людей в видео с использованием YOLO-v8. Проект анализирует видеопоток, детектирует людей, присваивает им уникальные ID и сохраняет результат в новом видеофайле с визуализацией. 

## 🛠️ Технологии
- OpenCV
- YOLO/PyTorch
- Docker

## 🎯 Выходные данные:
Файл rep_result.mp4 с:
- Обведенными людьми
- ID треков
- Уровнем уверенности модели

## ⚡ Быстрый старт

### 0. Установка Docker.
    Перед тем, как приступить к запуску программы, необходимо установить Docker на ПК.
    Для Windows: https://selectel.ru/blog/tutorials/docker-desktop-windows/;
    Для Linux: https://timeweb.com/ru/community/articles/kak-ustanovit-docker-na-ubuntu-22-04;
    Для MacOS: https://habr.com/ru/companies/evrone/articles/716630/.

### 1. Клонирование репозитория
```
git clone https://github.com/p1nt3r3st/working.git
```
### 2. Переходим в рабочую дирректорию
 ```
 cd working
 ```
 ### 3. Билдим проект
 ```
 docker build -t mynn .
 ```
 ### 4. Запускаем контейнер
 ```
 docker run -it --rm mynn
 ```
 --rm -  автоматически удаляет контейнер после завершения
