import numpy as np
from scipy.interpolate import interp1d

def replace_nan_and_inf_with_interpolation(array):
    """
    입력된 numpy 배열에서 각 열(column)별로 inf와 nan 값을 선형 보간으로 대체합니다.
    - 처음이나 끝에 inf 또는 nan이 있으면 가장 가까운 유효한 값을 사용합니다.
    - inf와 nan의 총 개수가 전체 데이터의 30%를 넘으면 원본 배열을 반환합니다.

    Parameters:
        array (numpy.ndarray): N x 3 형태의 2차원 numpy 배열.

    Returns:
        numpy.ndarray: inf와 nan이 선형 보간 또는 가장 가까운 값으로 대체된 배열.
    """
    # 입력 배열이 2차원인지 확인
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("Input array must be a 2D array with shape (N, 3).")

    # 복사본 생성 (원본 배열을 수정하지 않기 위해)
    result = array.copy()

    # 각 열(column)에 대해 처리
    for col in range(result.shape[1]):
        column_data = result[:, col]

        # inf와 nan 위치 확인
        invalid_mask = np.isnan(column_data) | np.isinf(column_data)

        # inf와 nan의 총 개수가 전체 데이터의 30%를 넘는 경우
        if np.sum(invalid_mask) > 0.3 * len(column_data):
            print(f"Too many invalid values in column {col} (more than 30%). Returning the original array.")
            return array  # 원본 배열 반환

        # 유효한 값의 인덱스와 값 추출
        valid_indices = np.where(~invalid_mask)[0]
        valid_values = column_data[~invalid_mask]

        # 유효한 값이 없으면 원본 배열 반환
        if len(valid_indices) == 0:
            raise ValueError(f"Column {col} contains only NaN or Inf values, cannot interpolate.")

        # 처음이나 끝에 inf 또는 nan이 있는 경우 가장 가까운 유효한 값으로 대체
        if invalid_mask[0]:  # 처음 값이 invalid인 경우
            column_data[0] = valid_values[0]
        if invalid_mask[-1]:  # 마지막 값이 invalid인 경우
            column_data[-1] = valid_values[-1]

        # 선형 보간 함수 생성
        interp_func = interp1d(valid_indices, valid_values, kind='linear', bounds_error=False, fill_value="extrapolate")

        # 보간을 통해 inf와 nan 값을 대체
        column_data[invalid_mask] = interp_func(np.where(invalid_mask)[0])

        # 결과를 다시 저장
        result[:, col] = column_data

    return result


# 예제 배열 (N x 3 형태)
data = np.array([
    [1.0, np.nan, 3.0],
    [2.0, 2.0, np.inf],
    [np.nan, 3.0, 5.0],
    [4.0, np.nan, 6.0],
    [5.0, 5.0, np.nan]
])

# 함수 호출
result = replace_nan_and_inf_with_interpolation(data)

print("Original Data:")
print(data)
print("\nProcessed Data:")
print(result)