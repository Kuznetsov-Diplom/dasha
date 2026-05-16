"""npbk.py — Нейросетевой Преобразователь Биометрия-Код (НПБК)

Реализация согласно ГОСТ Р 52633.0-2006, 52633.4-2011, 52633.5-2011
Глава 3 дипломной работы.
"""

import numpy as np
from typing import List, Dict, Any
from pathlib import Path
import json

class NPBKConverter:
    """Скелет НПБК по требованиям ГОСТ Р 52633.5-2011 (двухслойная нейросеть)."""
    
    def __init__(self, nbk_path: str = "models/nbk/nbk_model.json"):
        self.nbk_path = Path(nbk_path)
        self.nbk_path.parent.mkdir(parents=True, exist_ok=True)
        self.is_trained = False
        self.weights_layer1 = None
        self.weights_layer2 = None
        self.key_length = 256  # длина ключа в битах (по умолчанию)
    
    def register(self, own_vectors: List[np.ndarray], foreign_vectors: List[np.ndarray], 
                 key_length: int = 256) -> Dict[str, Any]:
        """Регистрация (обучение) преобразователя.
        own_vectors — векторы «Свой» (минимум 11 примеров по ГОСТ)
        foreign_vectors — векторы «Чужой» (большой объём)
        """
        self.key_length = key_length
        print(f"🔄 Обучение НПБК: {len(own_vectors)} «Свой» + {len(foreign_vectors)} «Чужой»")
        
        # TODO: Полная реализация по ГОСТ 52633.5
        # 1. Раздельное обучение каждого нейрона первого слоя
        # 2. Расчёт весов μ_i по формулам (2), (6) из стандарта
        # 3. Маскирование корреляционных связей
        # 4. Второй слой — коррекция ошибок
        
        self.is_trained = True
        # Сохранение параметров (упрощённо)
        self.save_nbk()
        return {"status": "success", "key_length": key_length, "message": "НПБК обучен"}
    
    def generate_key(self, biometric_vector: np.ndarray) -> str:
        """Генерация криптографического ключа из биометрического вектора."""
        if not self.is_trained:
            raise ValueError("НПБК ещё не обучен. Выполните register() сначала.")
        # TODO: Пропустить через два слоя нейронов
        # Вернуть битовую строку длиной key_length
        key = "0" * self.key_length  # placeholder
        print(f"🔑 Сгенерирован ключ длиной {self.key_length} бит")
        return key
    
    def verify(self, biometric_vector: np.ndarray) -> Dict[str, Any]:
        """Верификация: возвращает ключ + статус (Свой/Чужой)."""
        try:
            key = self.generate_key(biometric_vector)
            return {"key": key, "status": "success", "is_owner": True}
        except:
            return {"key": None, "status": "fail", "is_owner": False}
    
    def save_nbk(self):
        """Сохранение нейросетевого биометрического контейнера (НБК)."""
        data = {
            "key_length": self.key_length,
            "trained": self.is_trained,
            "description": "Упрощённый НБК по ГОСТ Р 52633.4"
        }
        with open(self.nbk_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"💾 НБК сохранён: {self.nbk_path}")
    
    def load_nbk(self):
        """Загрузка НБК."""
        if self.nbk_path.exists():
            with open(self.nbk_path, encoding="utf-8") as f:
                data = json.load(f)
            self.key_length = data.get("key_length", 256)
            self.is_trained = data.get("trained", False)
            print(f"✅ НБК загружен из {self.nbk_path}")


# Для тестирования
if __name__ == "__main__":
    nbk = NPBKConverter()
    print("✅ NPBKConverter создан успешно")
