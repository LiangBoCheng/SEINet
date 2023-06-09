import argparse
# EfficientNet-b0: [24, 40, 112, 320]
# EfficientNet-b3: [32, 48, 136, 384]
# EfficientNet-b4: [32, 56, 160, 448]
# EfficientNet-b5: [40, 64, 176, 512]
# EfficientNet-b6: [40, 72, 200, 576]
# EfficientNet-b7: [48, 80, 224, 640]
def getConfig():
    parser = argparse.ArgumentParser()

    # Model parameter settings
    parser.add_argument('--arch', type=str, default='7', help='Backbone Architecture')
    parser.add_argument('--channels', type=list, default=[48, 80, 224, 640])

    cfg = parser.parse_args()

    return cfg


if __name__ == '__main__':
    cfg = getConfig()
    cfg = vars(cfg)
    print(cfg)