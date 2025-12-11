import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- [가정] OpenVLA 모델이 뱉어낸 결과값 (예시 데이터) ---
# 실제 사용 시에는 model.predict_action()의 결과를 여기에 넣으세요.
# 예: action = action.detach().cpu().numpy()[0]
# 여기서는 테스트를 위해 임의의 값을 넣었습니다.
action = np.array([0.5, -0.2, 0.3, 0.0, 0.0, 0.1, 0.9]) 

def visualize_action(action_vector):
    """
    OpenVLA의 7-DoF 액션을 3D 화살표와 바 차트로 시각화합니다.
    """
    # 데이터 분리
    translation = action_vector[:3] # x, y, z
    rotation = action_vector[3:6]   # roll, pitch, yaw
    gripper = action_vector[6]      # 0(Open) ~ 1(Close)

    fig = plt.figure(figsize=(12, 5))
    fig.suptitle(f"OpenVLA Action Visualization", fontsize=16)

    # 1. 3D 이동 벡터 (Movement Vector)
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title("Translation Vector (XYZ)")
    
    # 원점(0,0,0)과 이동 목표점
    origin = [0, 0, 0]
    
    # 화살표 그리기 (검은색: 이동 방향)
    ax1.quiver(0, 0, 0, translation[0], translation[1], translation[2], 
               color='r', arrow_length_ratio=0.1, linewidth=2)
    
    # 원점 표시 (파란 점)
    ax1.scatter(0, 0, 0, color='b', s=50, label='Start')
    # 목표점 표시 (빨간 점)
    ax1.scatter(translation[0], translation[1], translation[2], color='r', s=50, label='Target')
    
    # 축 라벨 및 범위 설정
    ax1.set_xlabel('X (Forward/Back)')
    ax1.set_ylabel('Y (Left/Right)')
    ax1.set_zlabel('Z (Up/Down)')
    
    # 보기 좋게 축 범위 고정 (데이터 크기에 따라 조절)
    limit = max(np.abs(translation).max(), 0.5)
    ax1.set_xlim([-limit, limit])
    ax1.set_ylim([-limit, limit])
    ax1.set_zlim([-limit, limit])
    ax1.legend()

    # 2. 전체 값 바 차트 (Bar Chart)
    ax2 = fig.add_subplot(122)
    ax2.set_title("Action Values (Delta)")
    
    labels = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw', 'Gripper']
    colors = ['r', 'r', 'r', 'g', 'g', 'g', 'b'] # 위치, 회전, 그리퍼 색상 구분
    
    bars = ax2.bar(labels, action_vector, color=colors)
    ax2.axhline(0, color='black', linewidth=0.8) # 0점 기준선
    ax2.set_ylim([-1.1, 1.1]) # OpenVLA 출력은 보통 -1 ~ 1 사이 (그리퍼 제외)
    
    # 값 표시
    for bar in bars:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), 
                 ha='center', va='bottom' if yval > 0 else 'top')

    # 그리퍼 상태 텍스트 추가
    gripper_state = "CLOSED" if gripper > 0.5 else "OPEN"
    ax2.text(0.5, 0.9, f"Gripper Command: {gripper_state}", 
             transform=ax2.transAxes, ha='center', fontsize=12, 
             bbox=dict(facecolor='yellow', alpha=0.5))

    plt.tight_layout()
    
    # 파일로 저장 (SSH 환경 대비)
    plt.savefig("vla_result.png")
    print("-> 시각화 결과가 'vla_result.png'로 저장되었습니다.")
    
    # 화면 표시 (GUI 환경인 경우)
    try:
        plt.show()
    except:
        print("GUI 환경이 감지되지 않아 창을 띄우지 못했습니다. 저장된 파일을 확인하세요.")

# 실행
if __name__ == "__main__":
    visualize_action(action)