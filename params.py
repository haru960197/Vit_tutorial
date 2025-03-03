params_set = {
    # すべての基準となるパラメータ
    "normal": [
        {
            "patch_size": 8,
            "dim": 128,
            "depth": 3,
            "heads": 2,
            "mlp_dim": 256,
        },
    ],
    # パッチサイズ．画像サイズである32の約数である必要がある
    "patch_size": [
        {
            "patch_size": 2,
            "dim": 128,
            "depth": 3,
            "heads": 2,
            "mlp_dim": 256,
        },
        {
            "patch_size": 4,
            "dim": 128,
            "depth": 3,
            "heads": 2,
            "mlp_dim": 256,
        },
        {
            "patch_size": 16,
            "dim": 128,
            "depth": 3,
            "heads": 2,
            "mlp_dim": 256,
        },
    ],
    # 内部の次元
    "dim": [
        {
            "patch_size": 8,
            "dim": 256,
            "depth": 3,
            "heads": 2,
            "mlp_dim": 256,
        },
        {
            "patch_size": 8,
            "dim": 512,
            "depth": 3,
            "heads": 2,
            "mlp_dim": 256,
        },
        {
            "patch_size": 8,
            "dim": 1024,
            "depth": 3,
            "heads": 2,
            "mlp_dim": 256,
        },
    ],
    # レイヤー数
    "depth": [
        {
            "patch_size": 8,
            "dim": 128,
            "depth": 6,
            "heads": 2,
            "mlp_dim": 256,
        },
        {
            "patch_size": 8,
            "dim": 128,
            "depth": 12,
            "heads": 2,
            "mlp_dim": 256,
        },
        {
            "patch_size": 8,
            "dim": 128,
            "depth": 24,
            "heads": 2,
            "mlp_dim": 256,
        },
    ],
    # ヘッド数
    "heads": [
        {
            "patch_size": 8,
            "dim": 128,
            "depth": 3,
            "heads": 4,
            "mlp_dim": 256,
        },
        {
            "patch_size": 8,
            "dim": 128,
            "depth": 3,
            "heads": 8,
            "mlp_dim": 256,
        },
        {
            "patch_size": 8,
            "dim": 128,
            "depth": 3,
            "heads": 16,
            "mlp_dim": 256,
        },
    ],
    # MLP サイズ
    "mlp_dim": [
        {
            "patch_size": 8,
            "dim": 128,
            "depth": 3,
            "heads": 2,
            "mlp_dim": 512,
        },
        {
            "patch_size": 8,
            "dim": 128,
            "depth": 3,
            "heads": 2,
            "mlp_dim": 1024,
        },
        {
            "patch_size": 8,
            "dim": 128,
            "depth": 3,
            "heads": 2,
            "mlp_dim": 2048,
        },
    ],
}
