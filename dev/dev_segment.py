from shared.organelle_detector import segment_organelles

segment_organelles(
    "../data/Dataset01/raw/Mouse194/Cell1/3.dm4",
    "../test_train/prediction/20261028",
    rcnn_model_path="../test_train/model/20261027_141454test_1.pth",
    rcnn_config_path="../test_train/checkpoints/20261027_141454test_1/config.json",
)


# results = predict_bounding_boxes(
#     "../test_train/model/20261027_141454test_1.pth",
#     output_dir="../test_train/prediction/20261028",
#     config_path="../test_train/checkpoints/20261027_141454test_1/config.json",
#     image_paths=[
#         "../data/Dataset01/raw/Mouse194/Cell1/3.dm4",
#     ],
# )

# masks, scores = segment_sam2(
#     [
#         "../test_train/prediction/20261028/3_bbox.json",
#     ],
#     "../test_train/prediction/20261028",
# )


# # Convert the SAM masks into GeoJSON files for import back into QuPath
# def mask_to_geojson_feature(mask, label, object_id):
#     binary_mask = (mask.astype(np.uint8)) * 255
#     contours, _ = cv2.findContours(
#         binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
#     )

#     if not contours:
#         return None

#     largest_contour = max(contours, key=cv2.contourArea)
#     epsilon = 0.001 * cv2.arcLength(largest_contour, True)
#     approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)

#     coords = approx_contour.squeeze(axis=1).tolist()

#     if len(coords) < 3:
#         return None

#     if coords[0] != coords[-1]:
#         coords.append(coords[0])

#     feature = {
#         "type": "Feature",
#         "id": f"detection_{object_id}",
#         "geometry": {"type": "Polygon", "coordinates": [coords]},
#         "properties": {
#             "objectType": "annotation",
#         },
#     }
#     return feature


# # Build the full FeatureCollection
# features = []
# obj_idx = 0
# for idx, mask in enumerate(masks):
#     mask = masks[idx].squeeze()
#     label = "Object"

#     feature = mask_to_geojson_feature(mask, label, obj_idx)
#     if feature:
#         features.append(feature)
#         obj_idx += 1

# geojson_data = {"type": "FeatureCollection", "features": features}

# # Save to .geojson file
# output_dir = Path("../test_train/output")
# output_dir.mkdir(parents=True, exist_ok=True)

# output_geojson_path = output_dir / "test_annotations.geojson"
# with open(output_geojson_path, "w") as f:
#     json.dump(geojson_data, f, indent=2)

# print(f"GeoJSON exported to: {output_geojson_path}")

# from matplotlib import pyplot as plt

# plt.imshow(masks[10].squeeze())
# plt.show()
