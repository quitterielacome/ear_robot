from std_msgs.msg import Float32MultiArray  # top of file

self.pub_diag = self.create_publisher(Float32MultiArray, '/tracking_diag', 10)
