# SEINet
This project provides the code and results for 'Semantic-Edge Interactive Network for Salient Object Detection in Optical Remote Sensing Images', IEEE JSTARS, 2023.
[IEEE link](https://ieeexplore.ieee.org/document/10192474)
# Network Architecture

# Requirements
python 3.7 + pytorch 1.9.0
# Saliency maps
SEINet_EfficientNetB7_ORSI-4199 saliency maps: [Baidu](https://pan.baidu.com/s/1_3I-vXo91Mmd3Qp5FDobPw) (code:SEIN)  
SEINet_EfficientNetB7_EORSSD saliency maps: [Baidu](https://pan.baidu.com/s/1uiUu0TUS1hVXePrTlm9oEA) (code:SEIN)  
SEINet_EfficientNetB7_ORSSD saliency maps: [Baidu](https://pan.baidu.com/s/1PAabKvibGUEHaOgaoTf8Sw) (code:SEIN)  
SEINet_Res2Net50_ORSI-4199 saliency maps: [Baidu](https://pan.baidu.com/s/1ON6iDVAJpp1w6w9YZltGAw) (code:SEIN)  
SEINet_Res2Net50_EORSSD saliency maps: [Baidu](https://pan.baidu.com/s/12VCZsm3eLBw_9a7JTrDJbQ) (code:SEIN)  
SEINet_Res2Net50_ORSSD saliency maps: [Baidu](https://pan.baidu.com/s/1Nn6NpebLVpWLyt2jvXiJYg) (code:SEIN)  
SEINet_ResNet50_ORSI-4199 saliency maps: [Baidu](https://pan.baidu.com/s/1zyY-zeyeLwVIoFaxBKzdMw) (code:SEIN)  
SEINet_ResNet50_EORSSD saliency maps: [Baidu](https://pan.baidu.com/s/1XkoyZv_bqc3l__tvYRdmiA) (code:SEIN)  
SEINet_ResNet50_ORSSD saliency maps: [Baidu](https://pan.baidu.com/s/1GVtjELIvfQEtag-y0em1Qw) (code:SEIN)  
SEINet_VGG16_ORSI-4199 saliency maps: [Baidu](https://pan.baidu.com/s/1YTN2mrnhZaX4Q35GmWVg9g) (code:SEIN)  
SEINet_VGG16_EORSSD saliency maps: [Baidu](https://pan.baidu.com/s/1bbsd4MPbCRsq8wr2D9o-yA) (code:SEIN)  
SEINet_VGG16_ORSSD saliency maps: [Baidu](https://pan.baidu.com/s/115xLxnxZWyWc6Q_mWLflpQ) (code:SEIN)  
# Training
Run train_SEINet.py.  
For SEINet_VGG16, please modify paths of [VGG backbone](https://pan.baidu.com/s/1YBvqCHS-Y1JVIaW_rpSgLw) (code: SEIN) in /model/vgg.py.
# Pre-trained model and testing
Download the following pre-trained model and put them in ./models/, then run test_SEINet.py.  
[SEINet_EfficientNetB7_ORSI-4199](https://pan.baidu.com/s/11yJu1QsrbOFgfdoe8biYfg) (code:SEIN)   
[SEINet_EfficientNetB7_EORSSD](https://pan.baidu.com/s/18ESvcJ4AhiiqfloDv6UWZQ) (code:SEIN)   
[SEINet_EfficientNetB7_ORSSD](https://pan.baidu.com/s/1wXWW7UcblAdwO-NvuDOy6g) (code:SEIN)  
[SEINet_Res2Net50_ORSI-4199](https://pan.baidu.com/s/1LyeS4jh6Hy0SX-EwHgZyKg) (code:SEIN)  
[SEINet_Res2Net50_EORSSD](https://pan.baidu.com/s/1AFwK0avVqucSo61R4C6A8g) (code:SEIN)  
[SEINet_Res2Net50_ORSSD](https://pan.baidu.com/s/1UvPmLsbJolwnNit--iMPxQ) (code:SEIN)  
[SEINet_ResNet50_ORSI-4199](https://pan.baidu.com/s/1Aqo8_tiIfaPLvBT0Kk_tDg) (code:SEIN)  
[SEINet_ResNet50_EORSSD](https://pan.baidu.com/s/1Q5siaO9DkcyMa2Z6l7QejA) (code:SEIN)  
[SEINet_ResNet50_ORSSD](https://pan.baidu.com/s/1-mUwTMBNiwNSaYaHmNWJ3A) (code:SEIN)  
[SEINet_VGG16_ORSI-4199](https://pan.baidu.com/s/1vdIn2-RikWIBfkMoCPM6Cw) (code:SEIN)  
[SEINet_VGG16_EORSSD](https://pan.baidu.com/s/1G3amffaKLa5vv9Y7rda7ow) (code:SEIN)  
[SEINet_VGG16_ORSSD](https://pan.baidu.com/s/1mgZyXTFFmBzNbZt4V7g3Ww) (code:SEIN)  
# Evaluation Tool
You can use the [evaluation tool (MATLAB version)](https://github.com/MathLee/MatlabEvaluationTools) to evaluate the above saliency maps.
