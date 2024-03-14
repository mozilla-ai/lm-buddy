from lm_buddy.paths import LoadableAssetPath


def test_path_validation():
    abs_path = "/dogs/cats/chickens"
    print(LoadableAssetPath.model_validate(abs_path))
