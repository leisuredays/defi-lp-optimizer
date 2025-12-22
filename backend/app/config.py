"""
Configuration settings for the ML backend API

Loads environment variables and provides application configuration.
"""
import os
from typing import List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings:
    """Application settings"""

    # API Configuration
    API_VERSION: str = "1.0.0"
    API_TITLE: str = "Uniswap V3 ML Optimization API"
    API_DESCRIPTION: str = "Reinforcement Learning-based liquidity position optimization for Uniswap V3"

    # The Graph API
    GRAPH_API_KEY: str = os.getenv("GRAPH_API_KEY", "")

    # Model Configuration
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/ppo_uniswap_v3_v1.0.0.zip")
    MODEL_VERSION: str = "v1.0.0"

    # CORS Configuration
    CORS_ORIGINS: List[str] = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,https://defi-lab.xyz"
    ).split(",")

    # Server Configuration
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 8000))
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"

    # Request Limits
    MAX_DAYS_HISTORY: int = 365
    MIN_DAYS_HISTORY: int = 7
    DEFAULT_DAYS_HISTORY: int = 30

    # Gas Cost Estimates (USD)
    GAS_COSTS = {
        'ethereum': 100,
        'optimism': 3,
        'arbitrum': 3,
        'polygon': 2,
        'celo': 1
    }

    # Protocol Names
    PROTOCOL_NAMES = {
        0: 'ethereum',
        1: 'optimism',
        2: 'arbitrum',
        3: 'polygon',
        4: 'perpetual',
        5: 'celo'
    }

    def get_protocol_name(self, protocol_id: int) -> str:
        """Get protocol name from ID"""
        return self.PROTOCOL_NAMES.get(protocol_id, 'ethereum')

    def get_gas_cost(self, protocol_id: int) -> float:
        """Get gas cost estimate for protocol"""
        protocol_name = self.get_protocol_name(protocol_id)
        return self.GAS_COSTS.get(protocol_name, 100)


# Create global settings instance
settings = Settings()


# Validate critical settings on import
if not settings.GRAPH_API_KEY:
    print("⚠️  WARNING: GRAPH_API_KEY not set!")
    print("   Get your API key from: https://thegraph.com/studio/")
    print("   Add it to .env file: GRAPH_API_KEY=your_key_here")
