import numpy as np
import pinocchio as pin
import os


def create_dummy_urdf_file():
    """创建一个临时的URDF文件用于测试"""
    urdf_content = """<?xml version="1.0"?>
<robot name="simple_robot">
  <link name="base_link">
    <inertial>
      <mass value="0.001"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>

  <link name="link1">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0.5 0 0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>

  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-3.14" upper="3.14" effort="10" velocity="10"/>
  </joint>

  <link name="link2">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0.5 0 0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>

  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="link2"/>
    <origin xyz="1 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-3.14" upper="3.14" effort="10" velocity="10"/>
  </joint>
</robot>"""

    with open("simple_robot.urdf", "w") as f:
        f.write(urdf_content)
    return "simple_robot.urdf"


def load_from_urdf():
    """从URDF文件加载模型"""
    urdf_path = create_dummy_urdf_file()

    try:
        model = pin.buildModelFromUrdf(urdf_path)
        data = model.createData()

        print("从URDF成功加载模型!")
        print(f"自由度: {model.nq}")

        # 测试重力补偿
        q = np.array([0.5, 0.3])  # 两个关节的角度
        tau_gravity = pin.computeGeneralizedGravity(model, data, q)

        print(f"关节角度: {q}")
        print(f"重力补偿力矩: {tau_gravity}")

        return model, data

    except Exception as e:
        print(f"加载URDF失败: {e}")
        return None, None


if __name__ == "__main__":
    print("=== URDF测试 ===")
    load_from_urdf()