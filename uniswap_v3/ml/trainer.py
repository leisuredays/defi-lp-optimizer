"""
PPO Trainer for Uniswap V3 LP Optimization

Usage:
    python -m uniswap_v3.ml.trainer --data pool_data.parquet --steps 1000000
    python -m uniswap_v3.ml.trainer --resume models/ppo_latest
"""

import argparse
from pathlib import Path
from datetime import datetime
import json

import pandas as pd
import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from .environment import UniswapV3LPEnv
from .callbacks import RewardLoggingCallback, ProgressCallback


# 기본 설정
DEFAULT_CONFIG = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "policy_kwargs": {
        "net_arch": {"pi": [256, 256], "vf": [256, 256]},
        "activation_fn": torch.nn.Tanh,
    },
    "device": "auto",
}

DEFAULT_POOL_CONFIG = {
    "token0": {"symbol": "WETH", "decimals": 18},
    "token1": {"symbol": "USDT", "decimals": 6},
    "feeTier": 3000,
}


def load_data(data_path: Path, train_ratio: float = 0.8):
    """데이터 로드 및 분할"""
    if data_path.suffix == ".parquet":
        df = pd.read_parquet(data_path)
    elif data_path.suffix == ".csv":
        df = pd.read_csv(data_path)
    else:
        raise ValueError(f"지원하지 않는 파일 형식: {data_path.suffix}")

    df = df.sort_values("periodStartUnix").reset_index(drop=True)

    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy()
    eval_df = df.iloc[split_idx:].copy()

    print(f"📊 데이터 로드: {len(df)} rows")
    print(f"   Train: {len(train_df)}, Eval: {len(eval_df)}")

    return train_df, eval_df


def create_env(data: pd.DataFrame, pool_config: dict, investment: float = 10000):
    """환경 생성 - 전체 데이터를 1회 에피소드로 사용"""
    # 전체 데이터 길이를 에피소드 길이로 설정 (버퍼 10시간 제외)
    episode_length = len(data) - 10
    print(f"📈 Episode length: {episode_length} hours ({episode_length/24:.0f} days)")

    env = UniswapV3LPEnv(
        historical_data=data,
        pool_config=pool_config,
        initial_investment=investment,
        episode_length_hours=episode_length,
    )
    env = Monitor(env)
    return env


def train(
    data_path: Path,
    output_dir: Path,
    total_steps: int = 1_000_000,
    pool_config: dict = None,
    config: dict = None,
    resume_path: Path = None,
):
    """PPO 학습 실행"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 설정
    pool_config = pool_config or DEFAULT_POOL_CONFIG
    config = {**DEFAULT_CONFIG, **(config or {})}

    # 데이터 로드
    train_df, eval_df = load_data(data_path)

    # 환경 생성
    train_env = DummyVecEnv([lambda: create_env(train_df, pool_config)])

    # 모델 생성 또는 로드
    if resume_path and resume_path.exists():
        print(f"🔄 모델 로드: {resume_path}")
        model = PPO.load(resume_path, env=train_env)
    else:
        print("🆕 새 모델 생성")
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=config["learning_rate"],
            n_steps=config["n_steps"],
            batch_size=config["batch_size"],
            n_epochs=config["n_epochs"],
            gamma=config["gamma"],
            gae_lambda=config["gae_lambda"],
            clip_range=config["clip_range"],
            ent_coef=config["ent_coef"],
            vf_coef=config["vf_coef"],
            max_grad_norm=config["max_grad_norm"],
            policy_kwargs=config["policy_kwargs"],
            device=config["device"],
            verbose=1,
            tensorboard_log=str(output_dir / "logs"),
        )

    # 콜백 설정
    callbacks = CallbackList([
        RewardLoggingCallback(verbose=1),
        ProgressCallback(total_timesteps=total_steps, check_freq=10000),
        CheckpointCallback(
            save_freq=100_000,
            save_path=str(output_dir / "checkpoints"),
            name_prefix="ppo",
        ),
    ])

    # 학습
    print(f"\n🚀 학습 시작: {total_steps:,} steps")
    print(f"   Device: {model.device}")
    print(f"   Output: {output_dir}")

    model.learn(total_timesteps=total_steps, callback=callbacks, progress_bar=True)

    # 저장
    final_path = output_dir / "ppo_final"
    model.save(final_path)
    print(f"\n✅ 모델 저장: {final_path}")

    # 설정 저장
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump({
            "pool_config": pool_config,
            "training_config": {k: str(v) if not isinstance(v, (int, float, str, bool, list, dict)) else v
                               for k, v in config.items()},
            "total_steps": total_steps,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)

    return model


def main():
    parser = argparse.ArgumentParser(description="Uniswap V3 LP PPO Trainer")
    parser.add_argument("--data", type=Path, required=True, help="데이터 파일 경로 (.parquet/.csv)")
    parser.add_argument("--output", type=Path, default=Path("models"), help="출력 디렉토리")
    parser.add_argument("--steps", type=int, default=1_000_000, help="총 학습 스텝")
    parser.add_argument("--resume", type=Path, help="이어서 학습할 모델 경로")
    parser.add_argument("--lr", type=float, default=3e-4, help="학습률")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda)")

    args = parser.parse_args()

    config = {
        "learning_rate": args.lr,
        "device": args.device,
    }

    train(
        data_path=args.data,
        output_dir=args.output,
        total_steps=args.steps,
        config=config,
        resume_path=args.resume,
    )


if __name__ == "__main__":
    main()
