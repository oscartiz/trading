from eth_account import Account
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from loguru import logger

from config import settings


def build_clients() -> tuple[Info, Exchange]:
    """Return an Info (read) and Exchange (write) client pair."""
    wallet = Account.from_key(settings.private_key)
    info = Info(settings.base_url, skip_ws=True)
    exchange = Exchange(wallet, settings.base_url, account_address=settings.account_address)
    logger.info(
        "Hyperliquid clients ready | testnet={} address={}",
        settings.testnet,
        settings.account_address,
    )
    return info, exchange
