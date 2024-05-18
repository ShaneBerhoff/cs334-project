import models.arch.mobilenetv3_tl as mnv3tl
import models.arch.efficientnetv2b0_tl as env2b0tl
import models.arch.efficientnetv2b1_tl as env2b1tl
import models.arch.inceptionv3_tl as inv3tl
import models.arch.homebrew as homebrew

models = {
        "homebrew-etune": {
            "package": homebrew,
            "model": homebrew.Homebrew,
            "train": homebrew.train,
            "predict": homebrew.predict,
            "path": "homebrew-pipeline3-etune",
            "batch": 64,
            "epochs": 50,
            "epoch_tuning": True,
            "patience": 5,
            "n_mels": 128,
            "hop_len": 512
        },
        "homebrew": {
            "package": homebrew,
            "model": homebrew.Homebrew,
            "train": homebrew.train,
            "predict": homebrew.predict,
            "path": "homebrew-pipeline3",
            "batch": 64,
            "epochs": 50,
            "epoch_tuning": False,
            "patience": 5,
            "n_mels": 128,
            "hop_len": 512
        },
        "mnv3tl-etune": {
            "package": mnv3tl,
            "model": mnv3tl.MobileNetV3TL,
            "train": mnv3tl.train,
            "predict": mnv3tl.predict,
            "path": "mnv3tl-pipeline3-etune",
            "batch": 64,
            "epochs": 15,
            "epoch_tuning": True,
            "patience": 5,
            "n_mels": 224, # required dimension of 224x224
            "hop_len": 281 # from magic formula ((24414*(2618/1000))//(224-1)-5)
        },
        "mnv3tl": {
            "package": mnv3tl,
            "model": mnv3tl.MobileNetV3TL,
            "train": mnv3tl.train,
            "predict": mnv3tl.predict,
            "path": "mnv3tl-pipeline3",
            "batch": 64,
            "epochs": 15,
            "epoch_tuning": False,
            "patience": 5,
            "n_mels": 224, # required dimension of 224x224
            "hop_len": 281 # from magic formula ((24414*(2618/1000))//(224-1)-5)
        },
        "env2b0tl-etune": {
            "package": env2b0tl,
            "model": env2b0tl.EfficientNetV2B0TL,
            "train": env2b0tl.train,
            "predict": env2b0tl.predict,
            "path": "env2b0tl-pipeline3-etune",
            "batch": 64,
            "epochs": 18,
            "epoch_tuning": True,
            "patience": 5,
            "n_mels": 192, # required dimension of 192x192
            "hop_len": 328 # from magic formula ((24414*(2618/1000))//(192-1)-6)
        },
        "env2b0tl": {
            "package": env2b0tl,
            "model": env2b0tl.EfficientNetV2B0TL,
            "train": env2b0tl.train,
            "predict": env2b0tl.predict,
            "path": "env2b0tl-pipeline3",
            "batch": 64,
            "epochs": 18,
            "epoch_tuning": False,
            "patience": 5,
            "n_mels": 192, # required dimension of 192x192
            "hop_len": 328 # from magic formula ((24414*(2618/1000))//(192-1)-6)
        },
        "env2b1tl-etune": {
            "package": env2b1tl,
            "model": env2b1tl.EfficientNetV2B1TL,
            "train": env2b1tl.train,
            "predict": env2b1tl.predict,
            "path": "env2b1tl-pipeline3-etune",
            "batch": 64,
            "epochs": 11,
            "epoch_tuning": True,
            "patience": 5,
            "n_mels": 192, # required dimension of 192x192
            "hop_len": 328 # from magic formula ((24414*(2618/1000))//(192-1)-6)
        },
        "env2b1tl": {
            "package": env2b1tl,
            "model": env2b1tl.EfficientNetV2B1TL,
            "train": env2b1tl.train,
            "predict": env2b1tl.predict,
            "path": "env2b1tl-pipeline3",
            "batch": 64,
            "epochs": 11,
            "epoch_tuning": False,
            "patience": 5,
            "n_mels": 192, # required dimension of 192x192
            "hop_len": 328 # from magic formula ((24414*(2618/1000))//(192-1)-6)
        },
        "inv3tl-etune": {
            "package": inv3tl,
            "model": inv3tl.InceptionV3TL,
            "train": inv3tl.train,
            "predict": inv3tl.predict,
            "path": "inv3tl-pipeline3-etune",
            "batch": 32,
            "epochs": 15,
            "epoch_tuning": True,
            "patience": 5,
            "n_mels": 299, # required dimension of 299x299
            "hop_len": 211 # from magic formula ((24414*(2618/1000))//(299-1)-3)
        },
        "inv3tl": {
            "package": inv3tl,
            "model": inv3tl.InceptionV3TL,
            "train": inv3tl.train,
            "predict": inv3tl.predict,
            "path": "inv3tl-pipeline3",
            "batch": 32,
            "epochs": 15,
            "epoch_tuning": False,
            "patience": 5,
            "n_mels": 299, # required dimension of 299x299
            "hop_len": 211 # from magic formula ((24414*(2618/1000))//(299-1)-3)
        }
    }
