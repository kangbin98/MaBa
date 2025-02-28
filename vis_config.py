# config.py
class Config:
    # 路径配置
    IMAGES_DIR = "images"
    OUTPUT_ROOT = "Vis_result/Vis_image/test"
    MODEL_CHECKPOINT = "logs/CUHK-PEDES/20240917_101442_iira/best.pth"

    # 图像处理参数
    IMG_SIZE = (224, 224)
    MEAN = [0.48145466, 0.4578275, 0.40821073]
    STD = [0.26862954, 0.26130258, 0.27577711]

    # 模型参数
    PRETRAIN_CHOICE = "ViT-B/16"
    STRIDE_SIZE = 16
    TEMPERATURE = 0.07
    LOSS_NAMES = "itc+id+mlm"
    VOCAB_SIZE = 49408
    CMT_DEPTH = 2
    ID_LOSS_WEIGHT = 1.0
    MLM_LOSS_WEIGHT = 1.0

    # 注意力可视化参数
    TOKEN_CHOSEN = 86
    LAYERS_TO_HOOK = list(range(12))
    TARGET_SIZE = (224, 224)
    LOW_RES_SIZE = (14, 14)


# 实例化配置对象
cfg = Config()