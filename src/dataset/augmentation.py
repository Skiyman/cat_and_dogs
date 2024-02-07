import albumentations as A
from albumentations.pytorch import ToTensorV2


transform = A.Compose(
    [
        A.SmallestMaxSize(max_size=160), # Изменяем масштаб изображения так, чтобы минимальная сторона была равна max_size
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5), # Случайное аффинное преобразование
        A.RandomGamma(gamma_limit=(80, 120), eps=None, always_apply=False, p=0.5), # Изменяем гамму
        A.RandomCrop(height=128, width=128), # Вырезание в случайном месте
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5), # Случайное смещение цветов
        A.RandomBrightnessContrast(p=0.5), # Случайное измение контраста и яркости
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # Нормализация
        ToTensorV2(),
    ])
# Все это поможет увеличить объём данных

val_transforms = A.Compose(
    [
        A.SmallestMaxSize(max_size=160), # Изменяем масштаб изображения так, чтобы минимальная сторона была равна max_size
        A.CenterCrop(height=128, width=128), # Обрежем центральную часть (128X128)
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # Нормализация
        ToTensorV2(),
    ])
# В большенстве случаев важная часть находится в центре