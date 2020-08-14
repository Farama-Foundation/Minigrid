import os.path
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

class Resources:
    def get_image(self, image):
        return os.path.join(PROJECT_ROOT, "images", image)


@pytest.fixture("session")
def resources():
    return Resources()