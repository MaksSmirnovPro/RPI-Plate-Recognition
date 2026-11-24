
import cv2
import time
from picamera2 import Picamera2


def main():
    # Создание объекта камеры
    picam2 = Picamera2()

    # Настройка видеопотока
    config = picam2.create_video_configuration(
        main={
            "size": (1280, 720),
            "format": "BGR888"
        }
    )

    picam2.configure(config)

    # Запуск камеры
    picam2.start()

    # Небольшая задержка для инициализации камеры
    time.sleep(2)

    print("Камера запущена")
    print("Нажмите Q или ESC для выхода")

    try:
        while True:
            # Получение текущего кадра
            frame = picam2.capture_array()

            # Отображение изображения
            cv2.imshow("Raspberry Pi Camera - Live Preview", frame)

            # Обработка нажатия клавиш
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:
                break

    except KeyboardInterrupt:
        print("Остановка программы")

    finally:
        # Освобождение ресурсов
        picam2.stop()
        cv2.destroyAllWindows()
        print("Камера остановлена")


if __name__ == "__main__":
    main()
