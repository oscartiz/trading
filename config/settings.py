import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    private_key: str = os.environ["HL_PRIVATE_KEY"]
    account_address: str = os.environ["HL_ACCOUNT_ADDRESS"]
    testnet: bool = os.getenv("HL_TESTNET", "true").lower() == "true"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    @property
    def base_url(self) -> str:
        return "https://api.hyperliquid-testnet.xyz" if self.testnet else "https://api.hyperliquid.xyz"


settings = Settings()
