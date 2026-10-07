"""Native LightningCLI entry point for reproducible PathMNIST training.

Examples:
    python lightning_cli.py fit --config configs/lightning_pathmnist.yaml
    python lightning_cli.py test --config runs/.../best_config.yaml \
        --ckpt_path runs/.../blobs/agentic_model_v001.ckpt
"""

from __future__ import annotations

from collections.abc import Sequence


def main(args: Sequence[str] | None = None) -> int:
    try:
        from lightning.pytorch.cli import LightningCLI

        from lightning_components import PathMNISTDataModule, PathMNISTLitModule
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "LightningCLI dependencies are missing. Run "
            "`python -m pip install -r requirements.txt`."
        ) from exc

    LightningCLI(
        PathMNISTLitModule,
        PathMNISTDataModule,
        args=list(args) if args is not None else None,
        seed_everything_default=42,
        parser_kwargs={"default_env": True},
        save_config_kwargs={
            "config_filename": "resolved_lightning_config.yaml",
            "overwrite": False,
        },
        trainer_defaults={
            "deterministic": True,
            "devices": 1,
            "num_sanity_val_steps": 0,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
