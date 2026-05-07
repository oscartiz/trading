from eth_account import Account
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from loguru import logger

from config import settings


def build_clients(read_only: bool = False) -> tuple[Info, Exchange | None]:
    """Return an Info (read) and Exchange (write) client pair.

    With read_only=True the Exchange is skipped and no wallet credentials are
    required — useful for paper trading or pure data-collection runs.
    """
    info = Info(settings.base_url, skip_ws=True)
    if read_only:
        logger.info("Hyperliquid Info client ready | testnet={} (read-only)", settings.testnet)
        return info, None
    wallet = Account.from_key(settings.private_key)
    exchange = Exchange(wallet, settings.base_url, account_address=settings.account_address)
    logger.info(
        "Hyperliquid clients ready | testnet={} address={}",
        settings.testnet,
        settings.account_address,
    )
    return info, exchange
