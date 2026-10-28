from shared.organelle_detector import predict_bounding_boxes, segment_sam2

results = predict_bounding_boxes(
    "../test_train/model/20261027_141454test_1.pth",
    output_dir="../test_train/prediction/20261028",
    config_path="../test_train/checkpoints/20261027_141454test_1/config.json",
    image_paths=[
        "../data/Dataset01/raw/Mouse194/Cell1/3.dm4",
    ],
)

masks, scores = segment_sam2(
    [
        "../test_train/prediction/20261028/3_bbox.json",
    ],
    "../test_train/prediction/20261028",
)

from matplotlib import pyplot as plt

plt.imshow(masks[0].squeeze())
plt.show()
