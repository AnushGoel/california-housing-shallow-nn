"""Page registry and the context object that every page receives."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from ..artifacts import Bundle
from ..config import Settings
from ..service import PredictionService
from ..storage import RecordStore


@dataclass
class AppContext:
    bundle: Bundle
    service: PredictionService
    settings: Settings
    scenarios: RecordStore
    scores: RecordStore
    user_storage_label: str
    user_storage_is_cloud: bool
    storage_warning: Optional[str] = None


def pages() -> dict:
    from . import about, batch, diagnostics, experiments, explore, game, overview, playground, testing, tuning
    registry: dict[str, Callable[[AppContext], None]] = {
        "🛣️ Overview": overview.render, "🗺️ Explore the data": explore.render, "📈 Experiments": experiments.render,
        "🎛️ Tuning": tuning.render, "🧪 Testing": testing.render, "🔍 Diagnostics": diagnostics.render,
        "🏠 Try a block group": playground.render, "📦 Batch scoring and drift": batch.render,
        "🎯 Beat the model": game.render, "🧭 How it works": about.render,
    }
    return registry
