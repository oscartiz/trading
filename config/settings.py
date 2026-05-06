import os
from functools import cached_property

from dotenv import load_dotenv

load_dotenv()


def _require_env(name: str) -> str:
    try:
        return os.environ[name]
    except KeyError as exc:
        raise RuntimeError(
            f"{name} is not set — required for live execution. "
            "Backtests do not need this; only the live order/exchange clients do."
        ) from exc


class Settings:
    testnet: bool = os.getenv("HL_TESTNET", "true").lower() == "true"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    @cached_property
    def private_key(self) -> str:
        return _require_env("HL_PRIVATE_KEY")

    @cached_property
    def account_address(self) -> str:
        return _require_env("HL_ACCOUNT_ADDRESS")

    @property
    def base_url(self) -> str:
        return "https://api.hyperliquid-testnet.xyz" if self.testnet else "https://api.hyperliquid.xyz"


settings = Settings()
