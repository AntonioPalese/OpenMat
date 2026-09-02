import pytest

import openmat as om

requires_cuda = pytest.mark.skipif(
    not om.cuda_is_available(), reason="no CUDA device available"
)


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    """Runs a test on both backends, skipping cuda when there is no device."""
    if request.param == "cuda" and not om.cuda_is_available():
        pytest.skip("no CUDA device available")
    return request.param


def approx(a, b, tol=1e-4):
    return abs(a - b) < tol
