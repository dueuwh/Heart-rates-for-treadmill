import numpy as np
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints

# 상태 전이 함수
def fx(x, dt):
    # x = [px, py, vx, vy]
    F = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    return np.dot(F, x)

# 측정 함수
def hx(x):
    # 측정값은 위치 [px, py]만 포함
    return x[:2]

# 초기 상태
x0 = np.array([0, 0, 1, 1])  # 초기 위치 (0,0)과 초기 속도 (1,1)
P0 = np.eye(4) * 0.1  # 초기 공분산 행렬

# 시간 간격
dt = 1.0

# 시그마 포인트 생성
points = MerweScaledSigmaPoints(n=4, alpha=0.1, beta=2., kappa=0)

# UKF 객체 생성
ukf = UKF(dim_x=4, dim_z=2, fx=fx, hx=hx, dt=dt, points=points)
ukf.x = x0
ukf.P = P0
ukf.R = np.eye(2) * 0.1  # 측정 노이즈 공분산
ukf.Q = np.eye(4) * 0.1  # 프로세스 노이즈 공분산

# 측정값 (예: 실제 측정값을 사용해야 함)
measurements = [
    [1, 1],
    [2, 2],
    [3, 3],
    [4, 4],
    [5, 5]
]

# UKF를 사용하여 위치와 속도 보정
for z in measurements:
    ukf.predict()
    ukf.update(z)
    print(f"Updated state: {ukf.x}")

# 최종 상태 출력
print(f"Final state: {ukf.x}")
