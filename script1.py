from plot import get_diff_jpeg_df, get_guetzli_df

dataset1 = "/home/Haoran/distortion_rate/temp/kodak512"
get_guetzli_df(directory=dataset1, effective_bytes=False, write_files=True)
