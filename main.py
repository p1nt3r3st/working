import cv2
from ultralytics import YOLO

import pandas as pd
import numpy as np
import os
import subprocess

from tqdm import tqdm

import torch


def main():
    """Функция main отрисовывает на видео crowd.mp4 bounding box для людей.
     На выходе создаётся новый видеоролик rep_result.mp4"""

    path = 'crowd.mp4'
    model = YOLO('yolov8n.pt')
    dict_classes = model.model.names
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    def risize_frame(frame, scale_percent):
        """Функция изменяет размер изображения в процентном соотношении"""
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)

        resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        return resized

    def filter_tracks(centers, patience):
        """Функция для обновления хранимых кадров объекта.
        patience последних кадров будут храниться.
        """
        filter_dict = {}
        for k, i in centers.items():
            d_frames = i.items()
            filter_dict[k] = dict(list(d_frames)[-patience:])

        return filter_dict

    def update_tracking(centers_old, obj_center, thr_centers, lastKey, frame, frame_max):
        """Функция отслеживания траектории движения объектов. Обновляет координаты центра объекта или создаёт новый."""
        is_new = 0
        lastpos = [(k, list(center.keys())[-1], list(center.values())[-1]) for k, center in centers_old.items()]
        lastpos = [(i[0], i[2]) for i in lastpos if abs(i[1] - frame) <= frame_max]
        # Вычисляем расстояние одного и того же объекта на 2-ух соседних кадрах
        previous_pos = [(k, obj_center) for k, centers in lastpos if
                        (np.linalg.norm(np.array(centers) - np.array(obj_center)) < thr_centers)]
        # Если расстояние < threshold, то обновляем координаты центра
        if previous_pos:
            id_obj = previous_pos[0][0]
            centers_old[id_obj][frame] = obj_center
        # Иначе модель будет думать, что на изображении появился новый объект с новым ID
        else:
            if lastKey:
                last = lastKey.split('D')[1]
                id_obj = 'ID' + str(int(last) + 1)
            else:
                id_obj = 'ID0'

            is_new = 1
            centers_old[id_obj] = {frame: obj_center}
            lastKey = list(centers_old.keys())[-1]

        return centers_old, id_obj, is_new, lastKey

    # Конфигурации
    verbose = False  # Вывод дополнительной информации во время предсказания
    scale_percent = 100  # Процент масштабирования исходного кадра
    conf_level = 0.4  # Порог уверенности модели для детекции
    thr_centers = 40  # Порог расстояния между центрами (старые/новые позиции)
    frame_max = 15  # Макс. кадров до потери объекта
    patience = 100  # Макс. количество хранимых треков центров
    # -------------------------------------------------------

    # Чтение видео
    video = cv2.VideoCapture(path)

    # Настройки детекции
    class_IDS = [0]  # Классы объектов для детекции (0 - люди в YOLO)
    centers_old = {}  # Словарь для хранения предыдущих позиций объектов

    # Счётчики и служебные переменные
    obj_id = 0  # Идентификатор текущего объекта
    end = []  # Список завершённых треков
    frames_list = []  # Список кадров для обработки
    count_p = 0  # Счётчик людей
    lastKey = ''  # Последний обработанный ключ объекта
    print(f'[INFO] - Режим подробного вывода: {verbose}')

    # Получение параметров видео
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = video.get(cv2.CAP_PROP_FPS)
    print('[INFO] - Исходное разрешение: ', (width, height))

    # Масштабирование видео
    if scale_percent != 100:
        print('[INFO] - Внимание: изменение масштаба может вызвать ошибки!')
        width = int(width * scale_percent / 100)
        height = int(height * scale_percent / 100)
        print('[INFO] - Новое разрешение: ', (width, height))

    # Настройки выходного видео
    video_name = 'result.mp4'
    output_path = "rep_" + video_name  # Путь для результата
    tmp_output_path = "tmp_" + output_path  # Временный файл
    VIDEO_CODEC = "MP4V"  # Кодек для записи
    output_path = '/app/output/' + output_path
    tmp_output_path = '/app/output/' + tmp_output_path
    

    output_video = cv2.VideoWriter(tmp_output_path,
                                   cv2.VideoWriter_fourcc(*VIDEO_CODEC),
                                   fps, (width, height))

    # Основной цикл обработки
    # for i in tqdm(range(int(video.get(cv2.CAP_PROP_FRAME_COUNT)))):
    for i in tqdm(range(int(5 * fps))):

        # Захват кадра
        _, frame = video.read()

        # Предобработка кадра
        frame = risize_frame(frame, scale_percent)
        ROI = frame  # Область интереса (в данном случае весь кадр)

        if verbose:
            print('Текущий размер кадра: ', (frame.shape[1], frame.shape[0]))

        # Детекция объектов
        y_hat = model.predict(ROI, conf=conf_level, classes=class_IDS, device=device, verbose=False)

        # Извлечение результатов детекции
        boxes = y_hat[0].boxes.xyxy.cpu().numpy()  # Координаты рамок
        conf = y_hat[0].boxes.conf.cpu().numpy()  # Уверенность детекции
        classes = y_hat[0].boxes.cls.cpu().numpy()  # Классы объектов

        # Сохранение в DataFrame для удобства обработки
        boxes_data = y_hat[0].boxes.cpu().numpy().data
        positions_frame = pd.DataFrame(boxes_data, columns=['xmin', 'ymin', 'xmax', 'ymax', 'conf', 'class'])

        # Преобразование числовых меток классов в текстовые
        labels = [dict_classes[i] for i in classes]

        ### Обработка каждого обнаруженного человека ###
        for ix, row in enumerate(positions_frame.iterrows()):
            # Получение координат объекта
            xmin, ymin, xmax, ymax, confidence, category = row[1].astype('int')

            # Вычисление центра bounding box'а
            center_x, center_y = int(((xmax + xmin)) / 2), int((ymax + ymin) / 2)

            # Обновление трекинга
            centers_old, id_obj, is_new, lastKey = update_tracking(
                centers_old,
                (center_x, center_y),
                thr_centers,
                lastKey,
                i,
                frame_max
            )

            # Отрисовка bounding box'а и центра
            cv2.rectangle(ROI, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)  # Прямоугольник

            text = id_obj + ':' + str(np.round(conf[ix], 2))

            # Параметры текста
            font = cv2.FONT_HERSHEY_TRIPLEX
            font_scale = 0.8
            thickness = 1
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

            # Координаты фона для текста
            text_bg_x1 = xmin
            text_bg_y1 = ymin - text_size[1] - 5
            text_bg_x2 = xmin + text_size[0] + 5
            text_bg_y2 = ymin

            # Отрисовка фона для текста
            cv2.rectangle(
                ROI,
                (text_bg_x1, text_bg_y1),
                (text_bg_x2, text_bg_y2),
                (50, 50, 50),  # Темно-серый фон
                -1  # Заливка прямоугольника
            )

            # Отрисовка текста поверх фона
            cv2.putText(
                img=ROI,
                text=text,
                org=(xmin + 3, ymin - 10),
                fontFace=font,
                fontScale=font_scale,
                color=(255, 255, 255),
                thickness=thickness
            )

        # Очистка старых треков
        centers_old = filter_tracks(centers_old, patience)


        # Подготовка к сохранению
        overlay = frame.copy()
        frames_list.append(frame)  # Сохранение кадра
        output_video.write(frame)  # Запись в видео

    # Завершение работы с видео
    output_video.release()

    # Конвертация в конечный формат
    if os.path.exists(output_path):
        os.remove(output_path)

    subprocess.run([
        "ffmpeg",
        "-i", tmp_output_path,
        "-crf", "18",
        "-preset", "veryfast",
        "-hide_banner",
        "-loglevel", "error",
        "-vcodec", "libx264",
        output_path
    ])
    os.remove(tmp_output_path)  # Удаление временного файла


if __name__ == '__main__':
    main()
