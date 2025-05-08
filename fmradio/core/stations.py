"""Radio station management module."""

from dataclasses import dataclass
from typing import List, Optional

@dataclass
class RadioStation:
    """Represents a radio station with its frequency and name."""
    name: str
    frequency: float  # in MHz
    description: Optional[str] = None

    def __str__(self) -> str:
        return f"{self.name} ({self.frequency:.1f} MHz)"

class StationManager:
    """Manages radio stations and scanning functionality."""
    
    def __init__(self):
        self.stations: List[RadioStation] = []
        self._initialize_default_stations()

    def _initialize_default_stations(self):
        """Initialize with default stations."""
        self.stations = [
            RadioStation("Flaix FM", 104.1, "Catalan music station"),
            RadioStation("Catalunya Ràdio", 89.7, "Catalan public radio"),
            RadioStation("Ràdio 4", 103.8, "Spanish public radio"),
        ]

    def add_station(self, name: str, frequency: float, description: Optional[str] = None) -> None:
        """Add a new station to the list."""
        station = RadioStation(name, frequency, description)
        self.stations.append(station)
        self.stations.sort(key=lambda x: x.frequency)

    def get_station_by_frequency(self, frequency: float) -> Optional[RadioStation]:
        """Get a station by its frequency."""
        for station in self.stations:
            if abs(station.frequency - frequency) < 0.1:  # Allow small frequency differences
                return station
        return None

    def get_stations_in_range(self, min_freq: float, max_freq: float) -> List[RadioStation]:
        """Get all stations within a frequency range."""
        return [s for s in self.stations if min_freq <= s.frequency <= max_freq]

    def get_all_stations(self) -> List[RadioStation]:
        """Get all stations."""
        return sorted(self.stations, key=lambda x: x.frequency) 