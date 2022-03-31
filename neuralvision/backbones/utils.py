from typing import List, Tuple


def validate_backbone_features(
    output_feature_size: List[Tuple[int]],
    output_channels: List[int],
    output_features: list,
):
    for idx, feature in enumerate(output_features):
        out_channel = output_channels[idx]
        h, w = output_feature_size[idx]
        expected_shape = (out_channel, h, w)

        err_msg1 = f"Expected shape: {expected_shape}, \
                        got: {feature.shape[1:]} at output IDX: {idx}"
        assert feature.shape[1:] == expected_shape, err_msg1

    err_msg2 = f"Expected that the length of the outputted features to be: \
        {len(output_feature_size)}, but it was: {len(output_features)}"
    assert len(output_features) == len(output_feature_size), err_msg2
