# Tinyface-continuous-representation
Given a testing image of its raw size, by adjusting the target resolution, the network outputs the representation for the testing image in the target size. The representation output from a pretrained feature extractor is concatenated with a ratio of target size to the raw size of the input image, and fed into a two layer MLP.
### Feature
1. With a larger target size set in testing, the recognition performance is better. But this is only the case when the target size is in the range of training image anchor sizes.
2. During testing, the image fed into the feature extractor is resized to a fixed size. This fixed size is of a tradeoff between the noise introduced in upsampling and the trained kernel using 224 sized training dataset Vggface2. When resizing to 160, we experimentally found the best performance.
### Performance on tinyface dataset

Target resolution  | r1 | r5 | r10 | r20 | mAP 
------------------ |--- |--- | --- | --- | ---
Target=16  | 0.6362 | 0.7054 | 0.6027 | 0.7569 | 0.5617
Target=32  | 0.6566|0.7325|0.7604|0.7851|0.5867
Target=64  | **0.6617**|0.7387|0.7690|0.7907|0.5947
Target=128  | 0.6537|0.7352|0.7620|0.7856|0.5903
Target=160  | 0.6542|0.7355|0.7607|0.7854|0.5903
resnet output  | 0.6587|0.7309|0.7612|0.7821|0.5943

### Checkpoint
https://drive.google.com/file/d/1hbinvCdbmsVJIH18dXTTD3vSv5mVvVYY/view?usp=sharing
