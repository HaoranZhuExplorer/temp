from plot import get_diff_jpeg_df, get_guetzli_df
import os
import io
import time


time_start = time.time()
get_guetzli_df(directory="/home/Haoran/distortion_rate/temp/raw512", effective_bytes=False)
#get_diff_jpeg_df(directory="/home/Haoran/distortion_rate/temp/raw512", effective_bytes=False)
time_end = time.time()
duration = time_end - time_start
print("duration: "+str(duration)+" seconds")


