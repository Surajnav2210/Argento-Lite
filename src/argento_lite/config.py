"""Load and validate the YAML configuration files."""

from __future__ import annotations

from pathlib import Path

import yaml

from argento_lite.schemas import AppConfig, CompanyIdentityConfig, PolicyConfig, UniverseConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
CACHE_DIR = PROJECT_ROOT / "data" / "cache"


def _read_yaml(path: Path) -> dict:
    with path.open() as fh:
        data = yaml.safe_load(fh)
    return data or {}


def load_policy(path: Path = CONFIG_DIR / "policy.yaml") -> PolicyConfig:
    return PolicyConfig.model_validate(_read_yaml(path))


def load_universe(path: Path = CONFIG_DIR / "universe.yaml") -> UniverseConfig:
    return UniverseConfig.model_validate(_read_yaml(path))


def load_companies(path: Path = CONFIG_DIR / "company_aliases.yaml") -> CompanyIdentityConfig:
    raw = _read_yaml(path)
    raw["aliases"] = {str(k): v for k, v in (raw.get("aliases") or {}).items()}   # symbols like 005935
    raw["non_company"] = [str(s) for s in (raw.get("non_company") or [])]
    return CompanyIdentityConfig.model_validate(raw)


def load_config(config_dir: Path = CONFIG_DIR) -> AppConfig:
    """Load and validate policy.yaml, universe.yaml and company_aliases.yaml from ``config_dir``."""
    return AppConfig(
        policy=load_policy(config_dir / "policy.yaml"),
        universe=load_universe(config_dir / "universe.yaml"),
        companies=load_companies(config_dir / "company_aliases.yaml"),
    )
