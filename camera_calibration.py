import cv2
import numpy as np
import os

# 체커보드의 가로 및 세로 내부 코너 개수
checkerboard_size = (3, 3)

# 체커보드의 한 변의 길이 (임의의 단위)
checkerboard_square_size = 1.0  #자로 재서 재설정 해야함

# 이미지 파일들이 있는 디렉토리 경로
image_directory = "path/to/your/images"

# 체커보드 코너를 저장할 리스트
objpoints = []  # 3D 객체 포인트 (체커보드 코너의 실제 3D 좌표)
imgpoints = []  # 2D 이미지 포인트 (체커보드 코너의 이미지 상의 좌표)

# 3D 객체 포인트 생성 (체커보드 코너의 실제 3D 좌표)
objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard_size[0],
                       0:checkerboard_size[1]].T.reshape(-1, 2)
objp *= checkerboard_square_size

# 디렉토리에서 이미지 파일 읽기
for filename in os.listdir(image_directory):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # 이미지 파일 확장자 확인
        image_path = os.path.join(image_directory, filename)

        # 이미지 불러오기
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 체커보드 코너 찾기
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

# 캘리브레이션 수행
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,
                                                   gray.shape[::-1], None,
                                                   None)

# 캘리브레이션 결과 출력
print("Camera Matrix:")
print(mtx)
print("\nDistortion Coefficients:")
print(dist)
