import socket
import threading
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import ast
import numpy as np


class OscilloscopeServer:
    def __init__(self, host='localhost', port=12346, buffer_size=1000):
        self.host = host
        self.port = port
        self.buffer_size = buffer_size
        self.data_buffer = deque(maxlen=buffer_size)
        self.time_buffer = deque(maxlen=buffer_size)
        self.current_data = [0.0] * 6  # 存储6个数据通道的当前值
        self.running = True

        # 设置matplotlib
        self.fig, self.axes = plt.subplots(2, 3, figsize=(15, 8))
        self.fig.suptitle('实时手指关节角度示波器', fontsize=16)

        # 初始化6个子图
        self.lines = []
        titles = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        for i, ax in enumerate(self.axes.flat):
            line, = ax.plot([], [], 'b-')
            ax.set_title(titles[i])
            ax.set_ylim(-3.5, 3.5)  # 弧度范围
            ax.grid(True)
            self.lines.append(line)

        plt.tight_layout()

    def start_server(self):
        """启动服务器"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)

        print(f"示波器服务器启动在 {self.host}:{self.port}")
        print("等待客户端连接...")

        # 启动动画
        ani = animation.FuncAnimation(self.fig, self.update_plot, interval=50, blit=True)

        # 在单独的线程中处理连接
        server_thread = threading.Thread(target=self.accept_connections)
        server_thread.daemon = True
        server_thread.start()

        plt.show()

    def accept_connections(self):
        """接受客户端连接"""
        try:
            while self.running:
                client_socket, addr = self.server_socket.accept()
                print(f"客户端已连接: {addr}")
                client_thread = threading.Thread(target=self.handle_client, args=(client_socket,))
                client_thread.daemon = True
                client_thread.start()
        except Exception as e:
            print(f"服务器错误: {e}")

    def handle_client(self, client_socket):
        """处理客户端数据"""
        try:
            while self.running:
                data = client_socket.recv(1024).decode('utf-8')
                if not data:
                    break

                # 解析接收到的数据
                try:
                    # 数据格式: "[0.00, 1.23, 0.45, 0.00, -0.78, 0.00]"
                    data_list = ast.literal_eval(data)
                    if len(data_list) == 6:
                        self.current_data = data_list
                        current_time = len(self.time_buffer)

                        # 添加到缓冲区
                        self.time_buffer.append(current_time)
                        self.data_buffer.append(data_list.copy())

                        print(f"接收: {data_list}")

                except (ValueError, SyntaxError) as e:
                    print(f"数据解析错误: {e}, 原始数据: {data}")

        except ConnectionResetError:
            print("客户端连接断开")
        except Exception as e:
            print(f"处理客户端时出错: {e}")
        finally:
            client_socket.close()

    def update_plot(self, frame):
        """更新示波器显示"""
        if not self.data_buffer:
            return self.lines

        # 获取最近的数据
        recent_data = list(self.data_buffer)
        time_data = list(self.time_buffer)

        # 更新每个通道的图形
        for i in range(6):
            channel_data = [data[i] for data in recent_data]
            self.lines[i].set_data(time_data[-len(channel_data):], channel_data)

            # 动态调整x轴范围
            if time_data:
                self.axes.flat[i].set_xlim(max(0, time_data[-1] - self.buffer_size), time_data[-1] + 10)

        return self.lines

    def stop_server(self):
        """停止服务器"""
        self.running = False
        if hasattr(self, 'server_socket'):
            self.server_socket.close()
        print("示波器服务器已停止")


if __name__ == "__main__":
    # 创建并启动示波器服务器
    oscilloscope = OscilloscopeServer()

    try:
        oscilloscope.start_server()
    except KeyboardInterrupt:
        print("\n正在关闭服务器...")
    finally:
        oscilloscope.stop_server()