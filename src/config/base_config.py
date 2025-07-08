from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
import json
import os
from pathlib import Path


class BaseConfig(ABC):
    """
    Abstrakte Basisklasse für alle Konfigurationen.
    
    Bietet grundlegende Funktionalitäten für das Laden, Speichern und Validieren
    von Konfigurationen.
    """
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialisiert die Konfiguration.
        
        Args:
            config_dict: Optionales Dictionary mit Konfigurationswerten
        """
        self._config = config_dict or {}
        self._validate_config()
    
    @abstractmethod
    def _validate_config(self) -> None:
        """
        Validiert die Konfiguration.
        
        Muss in Subklassen implementiert werden.
        """
        pass
    
    @abstractmethod
    def get_default_config(self) -> Dict[str, Any]:
        """
        Gibt die Standard-Konfiguration zurück.
        
        Returns:
            Dictionary mit Standard-Konfigurationswerten
        """
        pass
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Gibt einen Konfigurationswert zurück.
        
        Args:
            key: Schlüssel der Konfiguration
            default: Standardwert falls Schlüssel nicht existiert
            
        Returns:
            Konfigurationswert
        """
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        Setzt einen Konfigurationswert.
        
        Args:
            key: Schlüssel der Konfiguration
            value: Zu setzender Wert
        """
        self._config[key] = value
        self._validate_config()
    
    def update(self, config_dict: Dict[str, Any]) -> None:
        """
        Aktualisiert die Konfiguration mit neuen Werten.
        
        Args:
            config_dict: Dictionary mit neuen Konfigurationswerten
        """
        self._config.update(config_dict)
        self._validate_config()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Konvertiert die Konfiguration zu einem Dictionary.
        
        Returns:
            Dictionary mit Konfigurationswerten
        """
        return self._config.copy()
    
    def save_to_file(self, file_path: str) -> None:
        """
        Speichert die Konfiguration in eine JSON-Datei.
        
        Args:
            file_path: Pfad zur Datei
        """
        try:
            # Verzeichnis erstellen falls nicht vorhanden
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            raise RuntimeError(f"Fehler beim Speichern der Konfiguration: {e}")
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'BaseConfig':
        """
        Lädt eine Konfiguration aus einer JSON-Datei.
        
        Args:
            file_path: Pfad zur Datei
            
        Returns:
            Konfigurationsobjekt
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Konfigurationsdatei nicht gefunden: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            
            return cls(config_dict)
            
        except Exception as e:
            raise RuntimeError(f"Fehler beim Laden der Konfiguration: {e}")
    
    def merge_with_defaults(self) -> None:
        """
        Ergänzt die Konfiguration mit Standardwerten für fehlende Schlüssel.
        """
        defaults = self.get_default_config()
        
        for key, value in defaults.items():
            if key not in self._config:
                self._config[key] = value
        
        self._validate_config()
    
    def get_nested(self, key_path: str, default: Any = None) -> Any:
        """
        Gibt einen verschachtelten Konfigurationswert zurück.
        
        Args:
            key_path: Pfad zum Schlüssel (z.B. "chunker.chunk_size")
            default: Standardwert falls Schlüssel nicht existiert
            
        Returns:
            Konfigurationswert
        """
        keys = key_path.split('.')
        current = self._config
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        
        return current
    
    def set_nested(self, key_path: str, value: Any) -> None:
        """
        Setzt einen verschachtelten Konfigurationswert.
        
        Args:
            key_path: Pfad zum Schlüssel (z.B. "chunker.chunk_size")
            value: Zu setzender Wert
        """
        keys = key_path.split('.')
        current = self._config
        
        # Bis zum vorletzten Schlüssel navigieren
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Letzten Schlüssel setzen
        current[keys[-1]] = value
        self._validate_config()
    
    def __getitem__(self, key: str) -> Any:
        return self._config[key]
    
    def __setitem__(self, key: str, value: Any) -> None:
        self.set(key, value)
    
    def __contains__(self, key: str) -> bool:
        return key in self._config
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}({json.dumps(self._config, indent=2)})"
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._config})" 