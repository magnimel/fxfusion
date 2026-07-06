import torch
from typing import Any

def _as_output_list(output: Any) -> list[torch.Tensor]:
    if isinstance(output, torch.Tensor):
        return [output]

    if isinstance(output, (tuple, list)):
        return list(output)

    raise TypeError(f"Unsupported output type: {type(output)}")


def compare_outputs(
    actual,
    expected,
    rtol: float = 1e-3,
    atol: float = 1e-3,
) -> tuple[bool, str]:
    actual_outputs = _as_output_list(actual)
    expected_outputs = _as_output_list(expected)

    if len(actual_outputs) != len(expected_outputs):
        return (
            False,
            f"output count mismatch — actual: {len(actual_outputs)}, "
            f"expected: {len(expected_outputs)}",
        )

    for i, (actual_tensor, expected_tensor) in enumerate(
        zip(actual_outputs, expected_outputs)
    ):
        actual_tensor = actual_tensor.detach().cpu()
        expected_tensor = expected_tensor.detach().cpu()

        if actual_tensor.shape != expected_tensor.shape:
            return (
                False,
                f"output {i} shape mismatch — "
                f"actual: {actual_tensor.shape}, expected: {expected_tensor.shape}",
            )

        if actual_tensor.dtype != expected_tensor.dtype:
            return (
                False,
                f"output {i} dtype mismatch — "
                f"actual: {actual_tensor.dtype}, expected: {expected_tensor.dtype}",
            )

        if not actual_tensor.is_floating_point():
            if torch.equal(actual_tensor, expected_tensor):
                continue

            return False, f"output {i} integer/bool mismatch"

        actual_nan_mask = torch.isnan(actual_tensor)
        expected_nan_mask = torch.isnan(expected_tensor)

        if not torch.equal(actual_nan_mask, expected_nan_mask):
            return (
                False,
                f"output {i} NaN position mismatch — "
                f"actual NaN count: {actual_nan_mask.sum().item()}, "
                f"expected NaN count: {expected_nan_mask.sum().item()}",
            )

        if actual_nan_mask.any():
            # compare only the non-NaN entries with allclose
            non_nan = ~actual_nan_mask
            if torch.allclose(actual_tensor[non_nan], expected_tensor[non_nan], rtol=rtol, atol=atol):
                continue
            
            diff = (actual_tensor[non_nan] - expected_tensor[non_nan]).abs()
            return (
                False,
                f"output {i} max diff (excluding matched NaNs): {diff.max().item():.6f}, "
                f"mean diff: {diff.mean().item():.6f}",
            )

        if torch.allclose(actual_tensor, expected_tensor, rtol=rtol, atol=atol):
            continue

        diff = (actual_tensor - expected_tensor).abs()
        return (
            False,
            f"output {i} max diff: {diff.max().item():.6f}, "
            f"mean diff: {diff.mean().item():.6f}",
        )

    return True, ""


def check_correctness(
    engine,
    model,
    inputs: list[torch.Tensor],
    rtol: float = 1e-3,
    atol: float = 1e-3,
) -> tuple[bool, str]:
    """
    Standard correctness helper for all runtime tests.
    """
    with torch.no_grad():
        actual = engine.run(inputs)
        expected = model(*inputs)

    return compare_outputs(actual, expected, rtol=rtol, atol=atol)