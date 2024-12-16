import ffmpeg

# Input video file or source
input_file = './source/highway.mp4'    # or './highway.mp4'

# Output RTSP server URL
output_url = 'rtsp://localhost:8554/stream'

# Use ffmpeg-python to stream the video
(
    ffmpeg
    .input(input_file, re=None)  # re=None plays the input at normal speed
    .output(output_url, f='rtsp', rtsp_transport='tcp', codec='copy')  # Stream over RTSP
    .run()
)