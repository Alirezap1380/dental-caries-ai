# Stage 1: tooth enumeration, with and without copies of the test images

DENTEX, CC-BY-NC-SA 4.0. torchvision Faster R-CNN (MobileNetV3-FPN, COCO pretrained), 12 epochs, identical settings for all three models. The split is over all 755 evaluation images by recovered patient (302 images held out as validation + test).

- **honest** (510 images): no image shares content or recovered patient with a held-out image. Scored test images in its training set: 0.
- **swap** (510 images, same size): honest with 113 random images replaced by the excluded ones, which include copies of the test images. Scored test images in its training set: 54. **swap − honest is the effect of training on copies at fixed training-set size: figure 1.**
- **naive** (623 images): the whole enumeration subset, as the benchmark intends. Scored test images in its training set: 54. naive − honest also includes 113 more training images, so it is confounded.

Scored on the 54 held-out test images that have a human enumeration. A tooth counts as correct when its FDI number is right AND its box overlaps the human box at IoU ≥ 0.5. Patient-grouped 95% bootstrap CIs; differences are paired (same images, same resamples).

### Leakage tripwire (> 0.95)

- ⚠ swap recall: 0.955 [0.942, 0.968] — **expected**: 54 of 54 scored test images are byte-identical copies inside its training set. This is the leak the experiment measures, not a result to report.
- ⚠ naive recall: 0.953 [0.940, 0.966] — **expected**: 54 of 54 scored test images are byte-identical copies inside its training set. This is the leak the experiment measures, not a result to report.
- ⚠ swap precision: 0.978 [0.968, 0.987] — **expected**: 54 of 54 scored test images are byte-identical copies inside its training set. This is the leak the experiment measures, not a result to report.
- ⚠ naive precision: 0.981 [0.973, 0.989] — **expected**: 54 of 54 scored test images are byte-identical copies inside its training set. This is the leak the experiment measures, not a result to report.

| | honest | swap | naive | swap − honest (copies, size-matched) | naive − honest (confounded) |
|---|---|---|---|---|---|
| recall | 0.907 [0.878, 0.932] | 0.955 [0.942, 0.968] | 0.953 [0.940, 0.966] | **0.049 [0.030, 0.070]** | 0.047 [0.028, 0.068] |
| precision | 0.941 [0.917, 0.962] | 0.978 [0.968, 0.987] | 0.981 [0.973, 0.989] | **0.037 [0.021, 0.054]** | 0.040 [0.022, 0.059] |

![Figure 1](figure1_enumeration.svg)

*Figure 1. Teeth found and correctly numbered, enumerators trained with (swap) and without (honest) byte-identical copies of the scored test images, at equal training-set size.*

Note on the patient split: patient recovery finds no repeat visits under either pixel thumbnails or ResNet-50 features (treated as a real null; see README), so held-out patients are held-out images.
